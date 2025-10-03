import numpy as np
import torch
import torch.optim as optim
from torchinfo.torchinfo import summary
from sklearn.metrics import mean_absolute_error
from functools import partial

from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model  = model
        self.scaler = scaler
        self.output_seq_len = optim_args['output_seq_len']
        self.print_model_structure = optim_args['print_model']

        # optimizer / schedulers
        self.lrate  =  optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        self.if_lr_scheduler    = optim_args['lr_schedule']
        self.lr_sche_steps      = optim_args['lr_sche_steps']
        self.lr_decay_ratio     = optim_args['lr_decay_ratio']

        # curriculum learning
        self.if_cl          = optim_args['if_cl']
        self.cl_steps       = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        self.warm_steps     = optim_args['warm_steps']

        self.optimizer      = optim.Adam(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        self.lr_scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio) if self.if_lr_scheduler else None
        
        self.loss   = masked_mae
        self.clip   = 5

    # ---------- helpers ----------
    def _inv_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Inverse scaling for both cases:
        - StandardScaler with .inverse_transform (speed datasets)
        - functools.partial(re_max_min_normalization, _max=..., _min=...) (flow datasets)
        Works on torch tensors (no numpy roundtrip).
        """
        # case 1: z-score scaler with inverse_transform on torch tensors
        if hasattr(self.scaler, "inverse_transform"):
            return self.scaler.inverse_transform(t)

        # case 2: partial with keywords (_max, _min)
        if isinstance(self.scaler, partial):
            kw = getattr(self.scaler, "keywords", {}) or {}
            _max = kw.get("_max", None)
            _min = kw.get("_min", None)
            if _max is None or _min is None:
                return t  # nothing to do
            # convert to torch tensors on same device/dtype with proper broadcasting
            device = t.device
            dtype  = t.dtype
            if not torch.is_tensor(_max):
                _max = torch.as_tensor(_max, device=device, dtype=dtype)
            else:
                _max = _max.to(device=device, dtype=dtype)
            if not torch.is_tensor(_min):
                _min = torch.as_tensor(_min, device=device, dtype=dtype)
            else:
                _min = _min.to(device=device, dtype=dtype)
            # re_max_min_normalization: (x+1)/2 * (_max-_min) + _min
            return (t + 1.0) / 2.0 * (_max - _min) + _min

        # default: return as-is
        return t

    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
            for _ in range(batch_num):
                if _ < self.warm_steps:
                    self.cl_len = self.output_seq_len
                elif _ == self.warm_steps:
                    self.cl_len = 1
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lrate
                else:
                    if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                        self.cl_len += int(self.if_cl)
            print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num'])==0:
            summary(self.model, input_data=input)
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                tmp = 1
                for _ in param.shape:
                    tmp = tmp*_
                parameter_num += tmp
            print("Parameter size: {0}".format(parameter_num))

    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        self.print_model(**kwargs)

        output  = self.model(input)                 # expect [B, T, N] or [B, T, N, 1]
        output  = output.transpose(1,2)             # -> [B, N, T]

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.lrate))
        else:
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        # ---- inverse-scale both predict and target (no direct _max/_min indexing) ----
        # for flow datasets: model outputs [B,N,T], we temporarily add channel dim to reuse same formula
        pred_scaled = self._inv_scale(output.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze(-1)   # [B,N,T]
        real_scaled = self._inv_scale(real_val.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze(-1) # [B,N,T]

        mae_loss = self.loss(pred_scaled[:, :self.cl_len, :], real_scaled[:, :self.cl_len, :])
        loss = mae_loss
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = masked_mape(pred_scaled, real_scaled, 0.0)
        rmse = masked_rmse(pred_scaled, real_scaled, 0.0)
        return mae_loss.item(), mape.item(), rmse.item()

    def eval(self, device, dataloader, model_name, **kwargs):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx   = data_reshaper(x, device)
            testy   = data_reshaper(y, device)
            output  = self.model(testx)         # [B,T,N]
            output  = output.transpose(1,2)     # [B,N,T]

            predict = self._inv_scale(output.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze(-1)
            real_val= self._inv_scale(testy.transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze(-1)

            loss = self.loss(predict, real_val, 0.0).item()
            mape = masked_mape(predict,real_val,0.0).item()
            rmse = masked_rmse(predict,real_val,0.0).item()

            print("test: {0}".format(loss), end='\r')

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss,mvalid_mape,mvalid_rmse

    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save=True, **kwargs):
        # helper to inverse scale (static context)
        def _inv_scale_static(t: torch.Tensor, scaler_obj):
            if hasattr(scaler_obj, "inverse_transform"):
                return scaler_obj.inverse_transform(t)
            if isinstance(scaler_obj, partial):
                kw = getattr(scaler_obj, "keywords", {}) or {}
                _max = kw.get("_max", None); _min = kw.get("_min", None)
                if _max is None or _min is None: return t
                device = t.device; dtype = t.dtype
                if not torch.is_tensor(_max): _max = torch.as_tensor(_max, device=device, dtype=dtype)
                else: _max = _max.to(device=device, dtype=dtype)
                if not torch.is_tensor(_min): _min = torch.as_tensor(_min, device=device, dtype=dtype)
                else: _min = _min.to(device=device, dtype=dtype)
                return (t + 1.0) / 2.0 * (_max - _min) + _min
            return t

        model.eval()
        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)  # [num, T, N]
        realy   = realy.transpose(1, 2)                          # [num, N, T]
        y_list  = []
        for itera, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx   = data_reshaper(x, device)
            testy   = data_reshaper(y, device).transpose(1, 2)   # [B,N,T]
            with torch.no_grad():
                preds   = model(testx)                           # [B,T,N]
            outputs.append(preds)
            y_list.append(testy)
        yhat    = torch.cat(outputs,dim=0)[:realy.size(0),...]   # [num,T,N]
        y_list  = torch.cat(y_list, dim=0)[:realy.size(0),...]   # [num,N,T]

        assert torch.where(y_list == realy)

        # inverse scale to original units
        realy_scaled = _inv_scale_static(realy.unsqueeze(-1), scaler).squeeze(-1)  # [num,N,T]
        yhat_scaled  = _inv_scale_static(yhat.transpose(1,2).unsqueeze(-1), scaler).transpose(1,2).squeeze(-1)  # [num,N,T]

        amae, amape, armse = [], [], []
        for i in range(12):
            pred = yhat_scaled[:,:,i]
            real = realy_scaled[:,:,i]
            if kwargs['dataset_name'] in ('PEMS04','PEMS08'):
                mae  = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()
                print(f'Evaluate best model on test data for horizon {i+1:02d}, Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}')
                amae.append(mae); amape.append(mape); armse.append(rmse)
            else:
                metrics = metric(pred, real)
                print(f'Evaluate best model on test data for horizon {i+1:02d}, Test MAE: {metrics[0]:.4f}, Test RMSE: {metrics[2]:.4f}, Test MAPE: {metrics[1]:.4f}')
                amae.append(metrics[0]); amape.append(metrics[1]); armse.append(metrics[2])

        print('(On average over 12 horizons) Test MAE: {:.2f} | Test RMSE: {:.2f} | Test MAPE: {:.2f}% |'.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))

        if save:
            save_model(model, save_path_resume)
