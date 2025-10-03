## model.model


import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate= EstimationGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer      = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer      = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)

    def forward(self, history_data: torch.Tensor, dynamic_graph: torch.Tensor, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        """decouple layer

        Args:
            history_data (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            node_embedding_u (torch.Parameter): node embedding E_u
            node_embedding_d (torch.Parameter): node embedding E_d
            time_in_day_feat (torch.Parameter): time embedding T_D
            day_in_week_feat (torch.Parameter): time embedding T_W

        Returns:
            torch.Tensor: the un decoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
        """

        gated_history_data  = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data, dynamic_graph=dynamic_graph, static_graph=static_graph)   
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)         
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden

# class D2STGNN(nn.Module):
#     def __init__(self, **model_args):
#         super().__init__()
#         # attributes
#         self._model_args = dict(model_args)
#         # num_feat را یک‌بار استخراج کنیم و به‌صورت صریح نگه داریم
#         self.num_feat = int(self._model_args.get('num_feat', 1))
#         self._hidden_dim    = model_args['num_hidden']
#         self._node_dim      = model_args['node_hidden']
#         self._forecast_dim  = 256
#         self._output_hidden = 512
#         self._output_dim    = model_args['seq_length']

#         self._num_nodes     = model_args['num_nodes']
#         self._k_s           = model_args['k_s']
#         self._k_t           = model_args['k_t']
#         self._num_layers    = 5

#         model_args['use_pre']   = False
#         model_args['dy_graph']  = True
#         model_args['sta_graph'] = True

#         self._model_args    = model_args

#         # start embedding layer
#         self.embedding      = nn.Linear(self._in_feat, self._hidden_dim)

#         # time embedding
#         self.T_i_D_emb  = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
#         self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

#         # Decoupled Spatial Temporal Layer
#         self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
#         for _ in range(self._num_layers - 1):
#             self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

#         # dynamic and static hidden graph constructor
#         if model_args['dy_graph']:
#             self.dynamic_graph_constructor  = DynamicGraphConstructor(**model_args)
        
#         # node embeddings
#         self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
#         self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

#         # output layer
#         self.out_fc_1   = nn.Linear(self._forecast_dim, self._output_hidden)
#         self.out_fc_2   = nn.Linear(self._output_hidden, model_args['gap'])

#         self.reset_parameter()

#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.node_emb_u)
#         nn.init.xavier_uniform_(self.node_emb_d)
#         nn.init.xavier_uniform_(self.T_i_D_emb)
#         nn.init.xavier_uniform_(self.D_i_W_emb)

class D2STGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
    
        # ---- نگه‌داشتن کانفیگ و استخراج پارامترهای کلیدی ----
        self._model_args = dict(model_args)  # کپی تا اصل ورودی mutate نشه
    
        # ابعاد ورودی/مخفی
        self.num_feat        = int(self._model_args.get('num_feat', 1))
        self._in_feat        = self.num_feat                    # فقط سیگنال‌های ترافیکی (بدون کانال‌های زمانی)
        self._in_feat_total  = self._in_feat + 2               # اگر جایی کلِ ویژگی‌ها لازم شد (با ۲ کانال زمانی)
        self._hidden_dim     = int(self._model_args.get('num_hidden', 32))
        self._node_dim       = int(self._model_args.get('node_hidden', 10))
        self._forecast_dim   = int(self._model_args.get('forecast_dim', 256))  # اگر در YAML نیست همان 256
        self._output_hidden  = int(self._model_args.get('output_hidden', 512)) # اگر در YAML نیست همان 512
        self._output_dim     = int(self._model_args.get('seq_length', 12))
        self._num_nodes      = int(self._model_args.get('num_nodes'))
        self._k_s            = int(self._model_args.get('k_s', 2))
        self._k_t            = int(self._model_args.get('k_t', 3))
        self._num_layers     = int(self._model_args.get('num_layers', 5))
        self.time_emb_dim    = int(self._model_args.get('time_emb_dim', 10))
        self.gap             = int(self._model_args.get('gap', 3))
    
        # فلگ‌ها (با مقدار پیش‌فرض امن)
        cfg = dict(self._model_args)
        cfg.setdefault('use_pre',   False)
        cfg.setdefault('dy_graph',  True)
        cfg.setdefault('sta_graph', True)
        self._model_args = cfg
    
        # ---- لایهٔ تع嵌یده‌سازی شروع (برای سیگنال‌های ترافیکی) ----
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
    
        # ---- امبدینگ‌های زمانی (به‌جای nn.Parameter، Embedding ایمن‌تر و کاراتر است) ----
        self.T_i_D_emb = nn.Embedding(288, self.time_emb_dim)  # time-in-day
        self.D_i_W_emb = nn.Embedding(7,   self.time_emb_dim)  # day-in-week
    
        # ---- لایه‌های Decoupled Spatial-Temporal ----
        self.layers = nn.ModuleList([
            DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **self._model_args)
        ])
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **self._model_args)
            )
    
        # ---- سازندهٔ گراف دینامیک (در صورت نیاز) ----
        if self._model_args.get('dy_graph', True):
            self.dynamic_graph_constructor = DynamicGraphConstructor(**self._model_args)
    
        # ---- امبدینگ‌های نود ----
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
    
        # ---- هد خروجی ----
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, self.gap)
    
        # ---- مقداردهی اولیهٔ همهٔ وزن‌ها ----
        self.reset_parameter()

    def reset_parameter(self):
        # embedding ورودی
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)

        # time embeddings
        nn.init.xavier_uniform_(self.T_i_D_emb.weight)
        nn.init.xavier_uniform_(self.D_i_W_emb.weight)

        # node embeddings
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)

        # output head
        nn.init.xavier_uniform_(self.out_fc_1.weight)
        if self.out_fc_1.bias is not None:
            nn.init.zeros_(self.out_fc_1.bias)
        nn.init.xavier_uniform_(self.out_fc_2.weight)
        if self.out_fc_2.bias is not None:
            nn.init.zeros_(self.out_fc_2.bias)

    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph   = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph   = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        """
        history_data: [B, L, N, F]  (F = num_feat [+ 2 time channels optionally])
        returns:
          history_data_trimmed [B,L,N,num_feat],
          node_emb_u [N,d], node_emb_d [N,d],
          time_in_day_feat [B,L,N,d], day_in_week_feat [B,L,N,d]
        """
        # --- resolve num_feat safely ---
        # اگر سازنده ست نکرده بود، ایمن‌سازی:
        if not hasattr(self, '_in_feat'):
            self._in_feat = getattr(self, 'num_feat', max(1, history_data.shape[-1] - 2))
        num_feat = getattr(self, 'num_feat', None)
        if num_feat is None:
            if hasattr(self, '_model_args') and isinstance(getattr(self, '_model_args'), dict):
                num_feat = int(self._model_args.get('num_feat', 1))
            else:
                # fallback: فرض می‌کنیم 2 کانال زمانی انتهایی هستند
                F_in = history_data.shape[-1]
                num_feat = max(1, F_in - 2) if F_in >= 3 else F_in
    
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
    
        B, L, N, F = history_data.shape
        device = history_data.device
    
        # --- build robust time indices ---
        if F <= num_feat:
            # no time channels provided -> use zeros
            tid_idx = torch.zeros((B, L, N), dtype=torch.long, device=device)      # 0..287
            diw_idx = torch.zeros((B, L, N), dtype=torch.long, device=device)      # 0..6
        else:
            # time_in_day in [0,1) -> *288 -> 0..287
            tid_raw = history_data[:, :, :, num_feat] * 288.0
            tid_idx = tid_raw.long().clamp_(0, 287)
            # day_in_week should be integer 0..6
            diw_raw = history_data[:, :, :, num_feat + 1]
            diw_idx = (diw_raw.long() % 7)
    
        # embeddings for time features
        time_in_day_feat = self.T_i_D_emb[tid_idx]  # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[diw_idx]  # [B, L, N, d]
    
        # keep only true traffic features
        history_data = history_data[:, :, :, :num_feat]
    
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat


    def forward(self, history_data):
        """Feed forward of D2STGNN.

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, N, L]
        """

        # ==================== Prepare Input Data ==================== #
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat   = self._prepare_inputs(history_data)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, history_data=history_data, time_in_day_feat=time_in_day_feat, day_in_week_feat=day_in_week_feat)

        # Start embedding layer
        history_data   = self.embedding(history_data)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(inh_backcast_seq_res, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        # Output Layer
        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden     = dif_forecast_hidden + inh_forecast_hidden
        
        # regression layer
        forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)

        return forecast
