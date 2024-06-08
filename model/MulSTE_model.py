import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Attention(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, attention_type = 'dot'):
        super(Attention, self).__init__()
        self._attention_type = attention_type
        self._input_dim = input_dim
        self._dim_q = hidden_dim
        self._dim_k = hidden_dim
        self._dim_v = hidden_dim
        
        # Linear layer to transform the query (decoder hidden state)
        self.query_fc = nn.Linear(self._input_dim, self._dim_q, bias=False)
        # Linear layer to transform the key (encoder hidden state)
        self.key_fc = nn.Linear(self._input_dim, self._dim_k, bias=False)
        # Linear layer to transform the value (encoder hidden state)
        self.value_fc = nn.Linear(self._input_dim, self._dim_v, bias=False)

        # Softmax layer to compute the attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, querys, keys, values):
        """
        querys: (batch_size, num_querys[num_nodes], input_dim)                                              
        keys: (batch_size, num_keys[num_nodes], input_dim)
        values: (batch_size, num_values[num_nodes], input_dim)
        
        dim_q == dim_k
        num_keys == num_values
        """
        # Transform the Q/K/V 
        # (batch_size, num_querys[num_nodes], input_dim) -> (batch_size, num_querys[num_nodes], dim_q) 
        Q = self.query_fc(querys)
        # (batch_size, num_keys[num_nodes], input_dim) -> (batch_size, num_keys[num_nodes], dim_k)
        K = self.key_fc(keys)
        # (batch_size, num_values[num_nodes], input_dim) -> (batch_size, num_values[num_nodes], dim_v)
        V = self.value_fc(values)
        
        assert Q.shape[-1] == K.shape[-1]
        assert K.shape[-2] == V.shape[-2]
        
        # Compute the attention weights
        if self._attention_type == 'dot':
            # dot product attention + scale
            # (batch_size, num_querys, num_keys)
            attention_weights = (Q @ K.transpose(-1,-2))/sqrt(self._dim_k)    
        elif self._attention_type == 'cosine':
            # cosine similarity attention
            # (batch_size, num_querys, dim_q)
            Q = Q / Q.norm(dim=-1, keepdim=True) 
            # (batch_size, num_keys, dim_k)
            K = K / K.norm(dim=-1, keepdim=True)
            # (batch_size, num_querys, num_keys)
            attention_weights = Q @ K.transpose(-1,-2)
        else:
            raise ValueError(f"Invalid attention type: {self._attention_type}")
        # (batch_size, num_querys, num_keys)                                                     
        attention_weights = self.softmax(attention_weights)
        # (batch_size, num_querys, num_keys) @ (batch_size, num_values, dim_v) -> (batch_size, num_querys, dim_v)
        attention_output = attention_weights @ V
        
        # (batch_size, num_querys, dim_v)
        return attention_output
   
class GCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: float = 0.0):                             
        super(GCN, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._bias_init_value = bias
        
        
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters() 
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)    
        
    def forward(self, adj: torch.Tensor, X: torch.Tensor):
        """
        adj: (num_nodes, num_nodes)
        X:   (batch_size, num_nodes, input_dim)
        
        return
        output: (batch_size, num_nodes, output_dim)
        """
        ## adj = calculate_laplacian_with_self_loop(adj)                                                
        output = adj @ X @ self.weights + self.biases
        output = torch.relu(output)                                                                     
        
        return output                                                                    


class MGAC(nn.Module):
    def __init__(self, M: int, selected_M: int, m_args_list: list, input_dim: int, single_output_dim: int, output_dim:int, ):                                        
        """
        M: The number of views
        selected_M: The number of selected views
        m_args_list: choose the needed m-th graph
        """
        super(MGAC, self).__init__()                                                                    
        # The number of views: M
        self._M = M 
        # The number of selected views: selected_M
        self._selected_M = selected_M
        self._m_args_list = m_args_list
        assert self._selected_M == len(self._m_args_list)
        
        ## self._adj = adj
        self._input_dim = input_dim
        self._single_output_dim = single_output_dim
        self._output_dim = output_dim
        
        # pre-defined graph
        self.GCNs = nn.ModuleList()
        for _ in range(0, self._M):
            self.GCNs.append(GCN(self._input_dim, self._single_output_dim))
        
        self._num_nodes = 282
        self._bias_init_value = 0.0
        # full adaptive graph
        self.full_node_embeddings = nn.Parameter(torch.randn(self._num_nodes, self._single_output_dim), requires_grad=True)
        self.full_weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._single_output_dim))                       
        self.full_biases = nn.Parameter(torch.FloatTensor(self._single_output_dim))
        # bias learning
        self.backcast_linear = nn.Linear(2 * self._single_output_dim, self._input_dim)
        self.bias_linear = nn.Linear(self._input_dim, self._single_output_dim)
        
        # attention weight_linear
        self.attention_weight_linear = nn.Linear(self._selected_M * self._single_output_dim, self._selected_M)
        
        # final linear 
        self.final_linear = nn.Linear(3 * self._single_output_dim, self._output_dim)
        
        # inital the node_embeddings
        self.reset_parameters()
        
        # self-attention (add the residual adaptive graph)
        self.self_attention = Attention(input_dim = (self._selected_M + 1) * self._single_output_dim, hidden_dim = self._single_output_dim, attention_type = 'dot')  # hidden_dim = self._output_dim 也可以，视情况调整
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.full_node_embeddings)
        nn.init.xavier_uniform_(self.full_weights)
        nn.init.constant_(self.full_biases, self._bias_init_value)  
    
    def forward(self, adj: torch.Tensor, X: torch.Tensor):
        """
        M-view graph adj: (m, num_nodes, num_nodes)
        X: (batch_size, num_nodes, input_dim)
        
        return
        output: (batch_size, num_nodes, output_dim)
        """    
        assert self._M == adj.shape[0]
        output_list = []
        pre_defined_output_list = []
        
        # pre-defined graph convolution
        for m in self._m_args_list:                                                                                    
            # output of m-th GCN: (batch_size, num_nodes, single_output_dim)
            output_m = self.GCNs[m](adj[m], X)                                                                  
            # output list: length = selected_M
            output_list.append(output_m)
            pre_defined_output_list.append(output_m)
        
        # pre_defined_output: (batch_size, num_nodes, selected_M * single_output_dim)    
        pre_defined_output = torch.cat(pre_defined_output_list, dim = -1)
        
        # attention_weight: (batch_size, num_nodes, selected_M)
        attention_weight = self.attention_weight_linear(pre_defined_output)
        attention_weight = F.softmax(attention_weight, dim = -1)
        
        # Adaptive Weighting: 
        Z_p = 0
        Sum = 0
        for i in range(self._selected_M):
            Sum = Sum + pre_defined_output[..., (i) * self._single_output_dim : (i+1) * self._single_output_dim] * attention_weight[..., (i) : (i+1)]
        Z_p = Sum
        
        # full adaptive graph convolution
        full_adaptive_adj = torch.softmax(torch.relu(torch.mm(self.full_node_embeddings, self.full_node_embeddings.transpose(0, 1))), dim=1)   
        
        full_adaptive_output = full_adaptive_adj @ X @ self.full_weights + self.full_biases
        
        # Z_g: (batch_size, num_nodes, 2 * single_output_dim)
        Z_g = torch.cat([Z_p, full_adaptive_output], dim = -1) 
            
        # backcast: (batch_size, num_nodes, input_dim)
        backcast = self.backcast_linear(Z_g)
        # residual: (batch_size, num_nodes, input_dim)
        residual = X - backcast
  
        Z_b = self.bias_linear(residual)          
               
        # final_output Z
        final_output = torch.cat([Z_g, Z_b], dim = -1)
        # (batch_size, num_nodes, 3*single_output_dim) -> (batch_size, num_nodes, output_dim)
        final_output = self.final_linear(final_output)
        ## final_output = torch.relu(final_output)
        final_output = final_output
        
        return final_output, torch.mean(attention_weight, dim = 0), torch.mm(self.full_node_embeddings, self.full_node_embeddings.transpose(0, 1)) 
                   
class MGACRULinear(nn.Module):
    def __init__(self, M: int, selected_M: int, m_args_list: list, input_dim: int, num_mgacru_units: int, output_dim: int, bias: float = 0.0): 
        """
        M: The number of views
        selected_M: The number of selected views
        m_args_list: choose the needed m-th graph
        """
        super(MGACRULinear, self).__init__()
        ## self._adj = adj                                                                                         
        # The number of views: M
        self._M = M 
        # The number of selected views: selected_M
        self._selected_M = selected_M
        self._m_args_list = m_args_list
        assert self._selected_M == len(self._m_args_list)
        
        self._input_dim = input_dim                                                                            
        self._num_mgacru_units = num_mgacru_units                                                                     
        self._output_dim = output_dim                                                                           
        self._bias_init_value = bias                                                                            

        
        self.mgac = MGAC(M = self._M, selected_M = self._selected_M, m_args_list = self._m_args_list, input_dim = self._input_dim + self._num_mgacru_units, single_output_dim = self._output_dim, output_dim = self._output_dim, )
        
        # "W" (input_dim + num_mgacru_units, output_dim)
        self.weights = nn.Parameter(torch.FloatTensor(self._output_dim, self._output_dim)) 
        # "b" (output_dim)
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))                                        
        self.reset_parameters()                                                                                 

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, adj: torch.Tensor, inputs: torch.Tensor, hidden_state: torch.Tensor):                    
        """
        M-view graph adj: (M, num_nodes, num_nodes)        
        inputs: (batch_size, num_nodes, input_dim) 
        hidden_state: (batch_size, num_nodes, num_mgacru_units) 
        
        return
        outputs: (batch_size, num_nodes, output_dim)
        """
        # print(inputs.shape)
        batch_size, num_nodes, input_dim = inputs.shape
        ## print(hidden_state.shape)
        _, _, num_mgacru_units = hidden_state.shape
        
        assert self._input_dim == input_dim
        # assert self._num_mgacru_units == num_mgacru_units
        
        # inputs "x" 
        # x (batch_size, num_nodes, input_dim)
        inputs = inputs.reshape((batch_size, num_nodes, self._input_dim))
        # hidden_state "h" 
        # h (batch_size, num_nodes, num_mgacru_units)
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_mgacru_units))
        # concatenation "[x, h]"
        # [x, h] (batch_size, num_nodes, input_dim + num_mgacru_units)
        concatenation = torch.cat((inputs, hidden_state), dim=-1)
        # support "mgac([x, h]|A)"
        # mgac([x, h]|A) (batch_size, num_nodes, output_dim)
        support, attention_weight, full_adaptive_adj = self.mgac(adj, concatenation) 
        # outputs "mgac([x, h]|A) W + b" 
        # mgac([x, h]|A) W + b (batch_size, num_nodes, output_dim)
        outputs = support @ self.weights + self.biases
        # return outputs
        return outputs, attention_weight, full_adaptive_adj

    def hyperparameters(self):
        return {
            "num_mgacru_units": self._num_mgacru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class MGACRUCell(nn.Module):
    def __init__(self, M: int, selected_M: int, m_args_list: list, input_dim: int, hidden_dim: int, ):
        """
        M: The number of views
        selected_M: The number of selected views
        m_args_list: choose the needed m-th graph
        """
        super(MGACRUCell, self).__init__()
        # The number of views: M
        self._M = M 
        # The number of selected views: selected_M
        self._selected_M = selected_M
        self._m_args_list = m_args_list
        assert self._selected_M == len(self._m_args_list)
        
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        
        # call MGACRULinear(input_dim = self._input_dim, num_mgacru_units = self._hidden_dim, output_dim = self._hidden_dim * 2, bias = 1.0)
        self.linear1 = MGACRULinear(self._M, self._selected_M, self._m_args_list, self._input_dim, self._hidden_dim, self._hidden_dim * 2, bias = 1.0, ) 
        # call MGACRULinear(input_dim = self._input_dim, num_mgacru_units = self._hidden_dim, output_dim = self._hidden_dim, bias = 0.0)
        self.linear2 = MGACRULinear(self._M, self._selected_M, self._m_args_list, self._input_dim, self._hidden_dim, self._hidden_dim, )               
        
    def forward(self, adj, inputs, hidden_state):
        """
        M-view graph adj: (M, num_nodes, num_nodes)        
        inputs: (batch_size, num_nodes, input_dim)
        hidden_state: (batch_size, num_nodes, hidden_dim)
        
        return
        new_hidden_state: (batch_size, num_nodes, hidden_dim)
        """
        # [r, u] = sigmoid([x, h]W + b)                                                                         
        # [r, u] (batch_size, num_nodes, (2 * hidden_dim))  
        concatenation = torch.sigmoid(self.linear1(adj, inputs, hidden_state)[0])
        attention_weight1, full_adaptive_adj1 = self.linear1(adj, inputs, hidden_state)[1], self.linear1(adj, inputs, hidden_state)[2]
        # r (batch_size, num_nodes, hidden_dim)
        # u (batch_size, num_nodes, hidden_dim)
        r, u = torch.chunk(concatenation, chunks=2, dim=-1)                                                      
        # c = tanh([x, (r * h)]W + b)
        # c (batch_size, num_nodes, hidden_dim)
        c = torch.tanh(self.linear2(adj, inputs, r * hidden_state)[0])
        attention_weight2, full_adaptive_adj2 = self.linear2(adj, inputs, hidden_state)[1], self.linear2(adj, inputs, hidden_state)[2]
        # h := u * h + (1 - u) * c                                                                              
        # h (batch_size, num_nodes * hidden_dim)
        new_hidden_state = u * hidden_state + (1 - u) * c
        # print(new_hidden_state.shape)
        return new_hidden_state, new_hidden_state, attention_weight1, full_adaptive_adj1, attention_weight2, full_adaptive_adj2

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class News_Bert(nn.Module):
    def __init__(self, hidden_dim: int, fine_tuned_bert_path: str):
        super(News_Bert, self).__init__() 
        
        self._hidden_dim = hidden_dim
        self._bert_hidden_size = 768
        self._num_nodes = 282
        # load the fine_tuned_bert
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self._bert_fine_tuned = torch.load(fine_tuned_bert_path, map_location='cuda:0')       
        self.news_linear = nn.Linear(self._bert_hidden_size, self._hidden_dim )
        
    def forward(self, daily_news_input_ids, daily_news_attention_mask, daily_news_token_type_ids, daily_valid_news_mask, i): 
        
        batch_size, feature_seq_len, num_news, num_tokens= daily_news_input_ids.shape
        # freeze    
        with torch.no_grad():
            output_bert = self._bert_fine_tuned.pretrained(input_ids = daily_news_input_ids[:,i,:,:].int().reshape((-1, num_tokens)), # (batch_size, num_news, num_tokens) -> (batch_size*num_news, num_tokens) # feature_seq_len = 1 
                                                            attention_mask = daily_news_attention_mask[:,i,:,:].int().reshape((-1, num_tokens)), # (batch_size, num_news, num_tokens) -> (batch_size* num_news, num_tokens)
                                                            token_type_ids = daily_news_token_type_ids[:,i,:,:].int().reshape((-1, num_tokens)))  # (batch_size, num_news, num_tokens) -> (batch_size* num_news, num_tokens)

            cls_last_hidden_state = output_bert.last_hidden_state[:, 0, :] # (batch_size*num_news, bert_hidden_size)
            cls_last_hidden_state_reshape = cls_last_hidden_state.reshape((batch_size, num_news, self._bert_hidden_size)) # (batch_size, num_news, bert_hidden_size)
        
        logits = self._bert_fine_tuned.fc(cls_last_hidden_state) # (batch_size*num_news, num_labels = 2)
        logits_reshape = logits.reshape((batch_size, num_news, 2)) # (batch_size, num_news, num_labels = 2)
        
        valid_logits = torch.where(daily_valid_news_mask[:,-1,:,:] > 0, logits_reshape[:,:,1:2], torch.tensor([0.]).type_as(logits_reshape))
        
        day_level_news_embedding = valid_logits * cls_last_hidden_state_reshape # (batch_size, num_news, bert_hidden_size)
         
        day_level_news_embedding = day_level_news_embedding.sum(dim = 1, keepdims = True) # (batch_size, 1, bert_hidden_size)
        textual_event_embedding = day_level_news_embedding # .sum(dim = 1) # (batch_size, 1, bert_hidden_size)          
        textual_event_embedding = self.news_linear(textual_event_embedding)    # (batch_size, 1, hidden_dim)

        # print(textual_event_embedding.shape)
        return textual_event_embedding

class Event_FC(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Event_FC, self).__init__()
        self._hidden_dim = hidden_dim
        self._num_nodes = 282
        # event_linear
        self.event_linear = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.event_d_linear = nn.Linear(self._hidden_dim, self._hidden_dim)
        self.event_s_linear = nn.Linear(self._hidden_dim, self._hidden_dim)
         
    def forward(self, adj, last_output_ab):
        event_d_embedding = self.event_d_linear(last_output_ab)
        event_s_embedding = self.event_s_linear(last_output_ab)
        
        return event_d_embedding, event_s_embedding
        
class Categorical_Event_Representation(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Categorical_Event_Representation, self).__init__()
        self._hidden_dim = hidden_dim
        self.embedding_day_of_week = nn.Embedding(8,self._hidden_dim)
        self.embedding_day_of_month = nn.Embedding(32,self._hidden_dim)
        self.embedding_day_of_year = nn.Embedding(367,self._hidden_dim)
        self.embedding_festival_type = nn.Embedding(4,self._hidden_dim)
        self._num_nodes = 282
        self.context_linear = nn.Linear(2*hidden_dim, hidden_dim) 
        self.active = nn.ReLU()
        
    def forward(self, inputs_festival,):
        """
        inputs_festival_feature: (batch_size, feature_seq_len = 7, input_dim = 4) 
        inputs_target_feature: (batch_size, target_seq_len = 5, input_dim = 4) 
        
        """
        
        day_of_week_x = self.embedding_day_of_week(inputs_festival[:,:,0])
        day_of_month_x = self.embedding_day_of_month(inputs_festival[:,:,1])
        day_of_year_x = self.embedding_day_of_year(inputs_festival[:,:,2])
        festival_type_x = self.embedding_festival_type(inputs_festival[:,:,3])
        categorical_event_embedding = day_of_week_x + day_of_month_x + day_of_year_x + festival_type_x
        
        return categorical_event_embedding.unsqueeze(-2)  # (batch_size, target_seq_len, input_dim) -> (batch_size, target_seq_len, 1, input_dim) 
        
class EGIA(nn.Module):
    def __init__(self, hidden_dim: int, self_attention_type = 'dot', cross_attention_type ='dot', ):
        super(EGIA, self).__init__()
        self._self_attention_type = self_attention_type
        self._cross_attention_type = cross_attention_type 
        
        self._self_input_dim = hidden_dim
        self._self_dim_q = hidden_dim
        self._self_dim_k = hidden_dim
        self._self_dim_v = hidden_dim
        
        self._cross_input_dim = hidden_dim
        self._cross_dim_q = hidden_dim
        self._cross_dim_k = hidden_dim
        self._cross_dim_v = hidden_dim
        
        # self-attention
        ## Linear layer to transform the query 
        self.self_query_fc = nn.Linear(self._self_input_dim, self._self_dim_q, bias=False)
        ## Linear layer to transform the key 
        self.self_key_fc = nn.Linear(self._self_input_dim, self._self_dim_k, bias=False)
        ## Linear layer to transform the value 
        self.self_value_fc = nn.Linear(self._self_input_dim, self._self_dim_v, bias=False)
        ## Linear layer to fuse the demand and supply【思考下此处】
        self.self_fuse_ds = nn.Linear(self._self_input_dim, self._self_dim_v, bias=False)
        
        # cross-attention
        ## Linear layer to transform the query 
        self.cross_query_fc = nn.Linear(self._cross_input_dim, self._cross_dim_q, bias=False)
        ## Linear layer to transform the key 
        self.cross_key_fc = nn.Linear(self._cross_input_dim, self._cross_dim_k, bias=False)
        ## Linear layer to transform the value
        self.cross_value_fc = nn.Linear(self._cross_input_dim, self._cross_dim_v, bias=False)

        # Softmax layer to compute the attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # event gate
        self.sigmoid = nn.Sigmoid()
        
        self.h_d_linear = nn.Linear(4*hidden_dim, hidden_dim)
        self.h_s_linear = nn.Linear(4*hidden_dim, hidden_dim)
        
    def forward(self, hidden_state_ab, hidden_state_d_inter, hidden_state_s_inter, ):
        """
        hidden_state_ab: (batch_size, num_nodes, hidden_dim)
        hidden_state_d_inter: (batch_size, num_nodes, hidden_dim)
        hidden_state_s_inter: (batch_size, num_nodes, hidden_dim)
        """
        # (batch_size, num_nodes, hidden_dim) -> (batch_size, num_nodes, 1, hidden_dim)
        hidden_state_ab = hidden_state_ab.unsqueeze(2)
        # (batch_size, num_nodes, hidden_dim) -> (batch_size, num_nodes, 1, hidden_dim)
        hidden_state_d_inter = hidden_state_d_inter.unsqueeze(2)
        # (batch_size, num_nodes, hidden_dim) -> (batch_size, num_nodes, 1, hidden_dim)
        hidden_state_s_inter = hidden_state_s_inter.unsqueeze(2)
        # (batch_size, num_nodes, 1, hidden_dim)
        hidden_state_ab_gated = self.sigmoid(hidden_state_ab)
        # (batch_size, num_nodes, 2, hidden_dim)
        hidden_state_ds_inter = torch.cat((hidden_state_d_inter, hidden_state_s_inter), dim = 2)
        
        
        # Self Attention
        # (batch_size, num_nodes, 2, hidden_dim)
        self_Q = self.self_query_fc(hidden_state_ds_inter)
        # (batch_size, num_nodes, 2, hidden_dim)
        self_K = self.self_key_fc(hidden_state_ds_inter)
        # (batch_size, num_nodes, 2, hidden_dim)
        self_V = self.self_value_fc(hidden_state_ds_inter)
        # Compute the attention weights
        if self._self_attention_type == 'dot':
            # dot product attention + scale
            # (batch_size, num_nodes, 2, 2)
            self_attention_weights = (self_Q @ self_K.transpose(-1,-2))/sqrt(self._self_dim_k)    
        elif self._self_attention_type == 'cosine':
            # cosine similarity attention
            # (batch_size, num_nodes, 2, hidden_dim)
            self_Q = self_Q / self_Q.norm(dim=-1, keepdim=True) 
            # (batch_size, num_nodes, 2, hidden_dim)
            self_K = self_K / self_K.norm(dim=-1, keepdim=True)
            # (batch_size, num_nodes, 2, 2)
            self_attention_weights = self_Q @ self_K.transpose(-1,-2)
        else:
            raise ValueError(f"Invalid self attention type: {self._self_attention_type}")
        # (batch_size, num_nodes, 2, 2)                                                    
        self_attention_weights = self.softmax(self_attention_weights)
        # (batch_size, num_nodes, 2, 2) @ (batch_size, num_nodes, 2, hidden_dim) -> (batch_size, num_nodes, 2, hidden_dim)
        self_attention_output = self_attention_weights @ self_V
        # chunk split: (batch_size, num_nodes, 2, hidden_dim) -> (batch_size, num_nodes, 1, hidden_dim) & (batch_size, num_nodes, 1, hidden_dim)
        hidden_state_d_inter, hidden_state_s_inter = torch.chunk(self_attention_output, chunks = 2, dim = 2) 
        
        
        # Cross Attention
        # (batch_size, num_nodes, 1, hidden_dim)
        cross_Q = self.cross_query_fc(hidden_state_ab)
        # (batch_size, num_nodes, 2, hidden_dim)
        cross_K = self.cross_key_fc(hidden_state_ds_inter)
        # (batch_size, num_nodes, 2, hidden_dim)
        cross_V = self.cross_value_fc(hidden_state_ds_inter)
        # Compute the attention weights
        if self._cross_attention_type == 'dot':
            # dot product attention + scale
            # (batch_size, num_nodes, 1, 2)
            cross_attention_weights = (cross_Q @ cross_K.transpose(-1,-2))/sqrt(self._cross_dim_k)    
        elif self._cross_attention_type == 'cosine':
            # cosine similarity attention
            # (batch_size, num_nodes, 1, hidden_dim)
            cross_Q = cross_Q / cross_Q.norm(dim=-1, keepdim=True)
            # (batch_size, num_nodes, 2, hidden_dim)
            cross_K = cross_K / cross_K.norm(dim=-1, keepdim=True)
            # (batch_size, num_nodes, 1, 2)
            cross_attention_weights = cross_Q @ cross_K.transpose(-1,-2)
        else:
            raise ValueError(f"Invalid cross attention type: {self._cross_attention_type}")
        # (batch_size, num_nodes, 1, 2)                                                     
        cross_attention_weights = self.softmax(cross_attention_weights)
        # (batch_size, num_nodes, 1, 2) @ (batch_size, num_nodes, 2, hidden_dim) -> (batch_size, num_nodes, 1, hidden_dim)
        cross_attention_output = cross_attention_weights @ cross_V
        
        hidden_state_d_inter = hidden_state_ab_gated * cross_attention_output + (1 - hidden_state_ab_gated) * hidden_state_d_inter
        ## hidden_state_d_inter = torch.cat((hidden_state_ab_gated  * cross_attention_output , (1 - hidden_state_ab_gated)* hidden_state_d_inter, hidden_state_ab, hidden_state_d_inter), dim = -1)
        ## hidden_state_d_inter = self.h_d_linear(hidden_state_d_inter)
        
        hidden_state_s_inter = hidden_state_ab_gated * cross_attention_output + (1 - hidden_state_ab_gated) * hidden_state_s_inter
        ## hidden_state_s_inter = torch.cat((hidden_state_ab_gated * cross_attention_output , (1 - hidden_state_ab_gated)* hidden_state_s_inter, hidden_state_ab, hidden_state_s_inter), dim = -1)
        ## hidden_state_s_inter = self.h_s_linear(hidden_state_s_inter)
        
        hidden_state_d_inter = hidden_state_d_inter.squeeze(2)
        hidden_state_s_inter = hidden_state_s_inter.squeeze(2)
        
        return hidden_state_d_inter, hidden_state_s_inter

class MulSTE(nn.Module):
    def __init__(self, M: int, selected_M_d: int, m_args_list_d: list, selected_M_s: int, m_args_list_s: list, selected_M_ab: int, m_args_list_ab: list, fine_tuned_bert_path, input_dim: int, hidden_dim: int, feature_seq_len: int, target_seq_len: int, with_or_without_interaction, with_or_without_event, **kwargs):                                              # **kwargs作用？
        """
        M: The number of views
        selected_M_d: The number of Demand prediction selected views
        m_args_list_d: choose the Demand prediction needed m-th graph
        selected_M_s: The number of Supply prediction selected views
        m_args_list_s: choose the Supply prediction needed m-th graph
        selected_M_ab: The number of Abnormal representation selected views
        m_args_list_ab: choose the Abnormal representation needed m-th graph
        bert_fine_tuned: The fine-tuned Bert, contains two blocks (1. self.pretrained, 2. self.fc).
        """
        super(MulSTE, self).__init__()        
        self._M = M 
        # The number of Demand prediction selected views: selected_M_d
        self._selected_M_d = selected_M_d
        self._m_args_list_d = m_args_list_d
        assert self._selected_M_d == len(self._m_args_list_d)
        # The number of Supply prediction selected views: selected_M_s
        self._selected_M_s = selected_M_s
        self._m_args_list_s = m_args_list_s
        assert self._selected_M_s == len(self._m_args_list_s)
        # The number of Abnormal representation selected views: selected_M_ab
        self._selected_M_ab = selected_M_ab
        self._m_args_list_ab = m_args_list_ab
        assert self._selected_M_ab == len(self._m_args_list_ab)
        
        self._input_dim = input_dim  
        self._hidden_dim = hidden_dim
        self._output_dim = 1
        
        # seq_len
        self._feature_seq_len = feature_seq_len
        self._target_seq_len = target_seq_len
        
        ## self._bert_hidden_size = 768
        
        # event representation
        # load the fine_tuned_bert
        self.news_bert = News_Bert(self._hidden_dim, fine_tuned_bert_path)
        self.event_fc = Event_FC(self._hidden_dim)
        
        # context representation
        self.categorical_event_representation = Categorical_Event_Representation(self._hidden_dim)
        
        # event gated fusion attention
        self.egia = EGIA(self._hidden_dim, 'cosine', 'cosine',)
        
        # MGACRU Cell
        self.mgacru_cell_d = MGACRUCell(self._M, self._selected_M_d, self._m_args_list_d, self._input_dim, self._hidden_dim)        
        self.mgacru_cell_d_inter = MGACRUCell(self._M, self._selected_M_d, self._m_args_list_d, self._input_dim, self._hidden_dim)  
        self.mgacru_cell_s = MGACRUCell(self._M, self._selected_M_s, self._m_args_list_s, self._input_dim, self._hidden_dim)        
        self.mgacru_cell_s_inter = MGACRUCell(self._M, self._selected_M_s, self._m_args_list_s, self._input_dim, self._hidden_dim)  
        
        if with_or_without_event == 'with_event' and with_or_without_interaction == 'with_interaction': 
            # with_abnormal & with_interaction
            self.mgacru_cell_ab = MGACRUCell(self._M, self._selected_M_ab, self._m_args_list_ab, self._input_dim, self._hidden_dim) 
            ## self.mgacru_cell_ab = MGACRUCell(self._M, self._selected_M_ab, self._m_args_list_ab, self._input_dim*2, self._hidden_dim) 
            
            self.h_d_linear = nn.Linear(1, hidden_dim)                                                       
            self.h_s_linear = nn.Linear(1, hidden_dim)                                                       
            self.h_a_linear = nn.Linear(2, hidden_dim) 
            
            # 【Fusion Output】 linear
            self.output_d_fusion_linear = nn.Linear(2*hidden_dim, target_seq_len)                                          
            self.output_s_fusion_linear = nn.Linear(2*hidden_dim, target_seq_len)                                           
            
            # 【Fusion Output】 conv
            self.output_d_fusion_conv = nn.Conv2d(2, target_seq_len * self._output_dim, kernel_size=(1, hidden_dim), bias=True) 
            self.output_s_fusion_conv = nn.Conv2d(2, target_seq_len * self._output_dim, kernel_size=(1, hidden_dim), bias=True)
            
        
    def forward(self, adj, inputs_d, inputs_s, daily_news_input_ids, daily_news_token_type_ids, daily_news_attention_mask, daily_valid_news_mask, inputs_outbreak, inputs_risk, inputs_abnormal_news_num, inputs_festival_feature, inputs_festival_target, with_or_without_interaction, with_or_without_event, ):
        """
        M-view graph adj: (M, num_nodes, num_nodes)
        inputs_d: (batch_size, feature_seq_len, num_nodes, input_dim) [feature_seq_len == num_timesteps_input]
        inputs_s: (batch_size, feature_seq_len, num_nodes, input_dim)
        with_or_without_interaction: (1,) 'with_interaction' or 'without_interaction'
        
        
        daily_news_input_ids: (batch_size, feature_seq_len, num_news, num_tokens)
        daily_news_token_type_ids: (batch_size, feature_seq_len, num_news, num_tokens)
        daily_news_attention_mask: (batch_size, feature_seq_len, num_news, num_tokens)
        daily_valid_news_mask: (batch_size, feature_seq_len, num_news, 1)
        
        inputs_outbreak: (batch_size, feature_seq_len, num_nodes, input_dim)
        inputs_risk: (batch_size, feature_seq_len, num_nodes, input_dim)
        inputs_abnormal_news_num: (batch_size, feature_seq_len, num_nodes, input_dim)
        
        with_or_without_event: (1,) 'with_event' or 'without_event'
        
        return
        last_output_d: (batch_size, num_nodes, 1)
        last_output_s: (batch_size, num_nodes, 1)                                                       
        """
        batch_size, feature_seq_len, num_nodes, input_dim_d = inputs_d.shape
        assert self._feature_seq_len == feature_seq_len
        
        _, _, _, input_dim_s = inputs_s.shape
        _, _, num_news, num_tokens= daily_news_input_ids.shape
        
        outputs_d = list()
        outputs_d_inter = list()
        outputs_s = list()
        outputs_s_inter = list()
        outputs_ab = list()
        
        # hidden_state_d "h_d" (batch_size, num_nodes, hidden_dim)
        # hidden_state_s "h_s" (batch_size, num_nodes, hidden_dim)
        hidden_state_d = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs_d)                 
        # hidden_state_d_inter = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs_d)
        hidden_state_s = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs_s)
        # hidden_state_s_inter = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs_s)
        hidden_state_ab = torch.zeros(batch_size, num_nodes, self._hidden_dim).type_as(inputs_outbreak)
        
        numerical_event = torch.cat((inputs_outbreak, inputs_risk), dim = -1) #(batch_size, num_nodes, 2)
        inputs_d = self.h_d_linear(inputs_d)
        inputs_s = self.h_s_linear(inputs_s)
        numerical_event_embedding = self.h_a_linear(numerical_event)
        
        # categorical_event_embedding (batch_size, num_nodes, hidden_dim)
        categorical_event_embedding = self.categorical_event_representation(inputs_festival = inputs_festival_feature)         
        # print(inputs_d.shape)
        # print(inputs_s.shape)
        # print(numerical_event.shape)
        inputs_d = inputs_d #+ categorical_event_embedding 
        inputs_s = inputs_s #+ categorical_event_embedding 
        st_align_event_embedding = numerical_event_embedding + categorical_event_embedding 

        for i in range(feature_seq_len):
            if with_or_without_event == 'with_event':               
                if with_or_without_interaction == 'with_interaction':
                    
                    if i == 0:

                        output_d, hidden_state_d = self.mgacru_cell_d(adj, inputs_d[:, i, :, :], hidden_state_d)[0:2]                          
                        attention_weight1_d, full_adaptive_adj1_d, attention_weight2_d, full_adaptive_adj2_d = self.mgacru_cell_d(adj, inputs_d[:, i, :, :], hidden_state_d)[2:6]
                        # print(hidden_state_d)
                        # print(hidden_state_d.shape)
                        ## output_d_inter, hidden_state_d_inter = self.mgacru_cell_d_inter(adj, inputs_d[:, i, :, :], torch.cat((torch.zeros(batch_size, num_nodes, 1).type_as(inputs_d), hidden_state_d_inter), dim=-1))                   
                        
                        ## print(hidden_state_s.shape)
                        output_s, hidden_state_s = self.mgacru_cell_s(adj, inputs_s[:, i, :, :], hidden_state_s)[0:2]                         
                        attention_weight1_s, full_adaptive_adj1_s, attention_weight2_s, full_adaptive_adj2_s = self.mgacru_cell_s(adj, inputs_s[:, i, :, :], hidden_state_s)[2:6]
                        ## print(hidden_state_s.shape)
                        ## output_s_inter, hidden_state_s_inter = self.mgacru_cell_s_inter(adj, inputs_s[:, i, :, :], torch.cat((torch.zeros(batch_size, num_nodes, 1).type_as(inputs_s), hidden_state_s_inter), dim=-1)) 
                        
                        ## print(hidden_state_ab.shape)
                        
                        # textual_event_embedding (batch_size, num_nodes, hidden_dim) from news_bert  
                        textual_event_embedding = self.news_bert(daily_news_input_ids = daily_news_input_ids, daily_news_attention_mask = daily_news_attention_mask, daily_news_token_type_ids = daily_news_token_type_ids, daily_valid_news_mask = daily_valid_news_mask, i = i)
                        
                        output_ab, hidden_state_ab = self.mgacru_cell_ab(adj, st_align_event_embedding[:, i, :, :] + textual_event_embedding, hidden_state_ab)[0:2]                                             
                        attention_weight1_ab, full_adaptive_adj1_ab, attention_weight2_ab, full_adaptive_adj2_ab = self.mgacru_cell_ab(adj, st_align_event_embedding[:, i, :, :] + textual_event_embedding, hidden_state_ab)[2:6]
                        ## print(hidden_state_ab.shape)
                        
                    elif i >= 1:
                        output_d, hidden_state_d = self.mgacru_cell_d(adj, inputs_d[:, i, :, :], inputs_d[:, i-1, :, :] + hidden_state_d)[0:2]                                                          
                        attention_weight1_d, full_adaptive_adj1_d, attention_weight2_d, full_adaptive_adj2_d = self.mgacru_cell_d(adj, inputs_d[:, i, :, :], inputs_d[:, i-1, :, :] + hidden_state_d)[2:6]
                        # print(hidden_state_d.shape)
                        ## output_d_inter, hidden_state_d_inter = self.mgacru_cell_d_inter(adj, inputs_d[:, i, :, :], torch.cat((inputs_d[:, i-1, :, :], hidden_state_d_inter), dim=-1))                                  
                        
                        output_s, hidden_state_s = self.mgacru_cell_s(adj, inputs_s[:, i, :, :], inputs_s[:, i-1, :, :] + hidden_state_s)[0:2]                                                           
                        attention_weight1_s, full_adaptive_adj1_s, attention_weight2_s, full_adaptive_adj2_s = self.mgacru_cell_s(adj, inputs_s[:, i, :, :], inputs_s[:, i-1, :, :] + hidden_state_s)[2:6]
                        ## output_s_inter, hidden_state_s_inter = self.mgacru_cell_s_inter(adj, inputs_s[:, i, :, :], torch.cat((inputs_s[:, i-1, :, :], hidden_state_s_inter), dim=-1))                                  
                        
                        # st_align_event_embedding_previous = torch.cat((inputs_outbreak[:, i-1, :, :], inputs_abnormal_news_num[:, i-1, :, :]), dim = -1)
                        # st_align_event_embedding_previous = self.h_a_linear(st_align_event_embedding_previous) 
                        
                        # textual_event_embedding (batch_size, num_nodes, hidden_dim) from news_bert  
                        textual_event_embedding = self.news_bert(daily_news_input_ids = daily_news_input_ids, daily_news_attention_mask = daily_news_attention_mask, daily_news_token_type_ids = daily_news_token_type_ids, daily_valid_news_mask = daily_valid_news_mask, i = i)                        
                        
                        output_ab, hidden_state_ab = self.mgacru_cell_ab(adj, st_align_event_embedding[:, i, :, :] + textual_event_embedding, st_align_event_embedding[:, i-1, :, :] + hidden_state_ab)[0:2]                                                            #【残差连接修改点4】         #  hidden_size + 2*input_dim
                        attention_weight1_ab, full_adaptive_adj1_ab, attention_weight2_ab, full_adaptive_adj2_ab = self.mgacru_cell_ab(adj, st_align_event_embedding[:, i, :, :] + textual_event_embedding, st_align_event_embedding[:, i-1, :, :] + hidden_state_ab)[2:6]
                        ## print(hidden_state_ab.shape)
                        
                    ## hidden_state_d_inter, hidden_state_s_inter = self.egia(hidden_state_ab, hidden_state_d_inter, hidden_state_s_inter)
                    hidden_state_d, hidden_state_s = self.egia(hidden_state_ab, hidden_state_d, hidden_state_s)
                    
                    outputs_d.append(output_d)
                    ## outputs_d_inter.append(output_d_inter)
                    outputs_s.append(output_s)
                    ## outputs_s_inter.append(output_s_inter) 
                    outputs_ab.append(output_ab)
            
            
        if with_or_without_event == 'with_event':    
            if with_or_without_interaction == 'with_interaction':
                
                # 【Decoder】
                # (1) last_output: 
                # last_output_ab "o_ab"
                last_output_ab = outputs_ab[-1]                 # (batch_size, num_nodes, hidden_dim)
                # last_output_d "o_d"       
                last_output_d = outputs_d[-1]                   # (batch_size, num_nodes, hidden_dim)
                # last_output_d_inter "o_d_inter"   
                ## last_output_d_inter = outputs_d_inter[-1]       # (batch_size, num_nodes, hidden_dim)
                # last_output_s "o_s" 
                last_output_s = outputs_s[-1]                   # (batch_size, num_nodes, hidden_dim)
                # last_output_s_inter "o_s_inter"  
                ## last_output_s_inter = outputs_s_inter[-1]
                              
                event_d_embedding, event_s_embedding = self.event_fc(adj, last_output_ab)
                
                # 【Fusion Output】linear 
                last_output_d = self.output_d_fusion_linear(torch.cat((event_d_embedding, last_output_d), dim=-1))  # (batch_size, num_nodes, 4*hidden_dim)    
                last_output_d = last_output_d.unsqueeze(-1).permute(0,2,1,3)
                last_output_s = self.output_s_fusion_linear(torch.cat((event_s_embedding, last_output_s), dim=-1))  #                
                last_output_s = last_output_s.unsqueeze(-1).permute(0,2,1,3)
                
        return last_output_d, last_output_s, attention_weight1_d, full_adaptive_adj1_d, attention_weight2_d, full_adaptive_adj2_d, attention_weight1_s, full_adaptive_adj1_s, attention_weight2_s, full_adaptive_adj2_s, attention_weight1_ab, full_adaptive_adj1_ab, attention_weight2_ab, full_adaptive_adj2_ab
                                                                                                    

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
    
