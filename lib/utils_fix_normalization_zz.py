import pandas as pd
import numpy as np
import torch

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def generate_dataset(X, args, normalizer):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: 
        - Node features of shape 
          (num_timesteps, num_vertices, num_features) or (num_timesteps, ..., num_features)
    :num_timesteps_input
    :num_timesteps_output
    :normalizer:
        - {'max01','max11','std','None'}
    
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_timesteps_input, num_vertices, num_features)  or (num_timesteps, num_timesteps_input, ..., num_features).
        - Node targets for the samples. Shape is
          (num_samples, num_timesteps_output, num_vertices, num_features) or (num_timesteps, num_timesteps_output, ..., num_features).
        - scaler 
    """
    
    # X -> torch.Tensor(torch.float32)
    if type(X) == torch.Tensor:
        # np.float64 -> torch.float32
        X = X.float()
    elif type(X) == np.ndarray:
        # np.ndarray -> torch.float32
        X = torch.FloatTensor(X)
 
    num_timesteps_input = args.T
    num_timesteps_output = args.l
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    intervals = [(i, i + (num_timesteps_input + num_timesteps_output)) for i in range(X.shape[0] - (num_timesteps_input + num_timesteps_output) + 1)] 

    # Save samples
    features, target = [], []
    for i, j in intervals:
        features.append(X[i: i + num_timesteps_input, ...]) 
        target.append(X[i + num_timesteps_input: j, ...])

    return torch.stack(features), torch.stack(target)

def generate_dataset_for_each_datatype(args):
    
    print("-------------------------------------------------------------")
    print("Generate demand-supply sequences...")
    """
    ds_matrix_t: (total_seq_len, num_vertices, num_features)
    ds_feature: (num_samples, num_timesteps_input, num_vertices, num_features)
    ds_target: (num_samples, num_timesteps_output, num_vertices, num_features)
    scaler: 
    """
    d_matrix = np.load(args.d_path)
    s_matrix = np.load(args.s_path)
    ds_matrix = np.load(args.ds_path)
    ds_matrix_t = ds_matrix.transpose((2,1,0))
    ds_feature, ds_target =  generate_dataset(ds_matrix_t, args,'std') 
    
    print("-------------------------------------------------------------")
    print( "Generate numerical event sequences...")
    """
    [Indicator1] outbreak_matrix_t: (total_seq_len, num_vertices, num_features)
    outbreak_feature: (num_samples, num_timesteps_output, num_vertices, num_features)
    """
    outbreak_matrix = np.load(args.outbreak_path)
    outbreak_matrix = outbreak_matrix[..., np.newaxis]
    outbreak_matrix_t = outbreak_matrix.transpose((1,0,2))
    outbreak_feature,_ = generate_dataset(outbreak_matrix_t, args, 'std')

    """
    [Indicator2] risk_matrix_t:
    """
    risk_matrix = np.load(args.risk_path)
    risk_matrix = risk_matrix[..., np.newaxis]
    risk_matrix_t = risk_matrix.transpose((1,0,2))
    risk_feature,_ = generate_dataset(risk_matrix_t, args, 'std') 
    risk_feature = torch.where(torch.isnan(risk_feature), torch.full_like(risk_feature, 0), risk_feature)
    
    """
    [Indicator3] abnormal_news_num_matrix
    """
    daily_news_text_labeled_full_date = pd.read_csv(args.daily_news_text_labeled_full_date_path)
    del daily_news_text_labeled_full_date['Unnamed: 0']
    abnormal_news_num_matrix = daily_news_text_labeled_full_date.groupby('date')['标题是否疫情相关'].sum()
    abnormal_news_num_matrix = abnormal_news_num_matrix.values
    abnormal_news_num_matrix = abnormal_news_num_matrix[:,np.newaxis, np.newaxis]
    abnormal_news_num_matrix = torch.Tensor(abnormal_news_num_matrix)
    abnormal_news_num_matrix = abnormal_news_num_matrix.repeat(1, 71, 1)
    abnormal_news_num_feature,_ = generate_dataset(abnormal_news_num_matrix, args, 'std') 
    abnormal_news_num_feature = torch.where(torch.isnan(abnormal_news_num_feature), torch.full_like(abnormal_news_num_feature, 0), abnormal_news_num_feature)
    
    print("-------------------------------------------------------------")
    print( "Generate categorical event sequences...")
    """
    festival_matrix
    """

    festival_matrix = np.load(args.festival_path)
    festival_matrix = np.insert(festival_matrix, obj = 7, values = 100, axis = 1)
    festival_matrix[:,7][(festival_matrix[:,4] == 0) & (festival_matrix[:,6] == 0)] = 0
    festival_matrix[:,7][(festival_matrix[:,4] == 1) & (festival_matrix[:,6] == 0)] = 1
    festival_matrix[:,7][(festival_matrix[:,4] == 0) & (festival_matrix[:,6] == 1)] = 2
    festival_matrix[:,7][(festival_matrix[:,4] == 1) & (festival_matrix[:,6] == 1)] = 3
    festival_matrix = festival_matrix[:,[0,1,2,-1]]

    festival_feature,festival_target = generate_dataset(festival_matrix, args, 'None')

    festival_feature = torch.where(torch.isnan(festival_feature), torch.full_like(festival_feature, 0), festival_feature)
    festival_target = torch.where(torch.isnan(festival_target), torch.full_like(festival_target, 0), festival_target)
    festival_feature = festival_feature.long()
    festival_target = festival_target.long()
    
    print("-------------------------------------------------------------")
    print( "Generate fine-tuning news...")
    """
    news_matrix_fine_tuning
    """
    news_matrix_fine_tuning = np.load(file=args.news_fine_tuning_path, allow_pickle=True)
    
    print("-------------------------------------------------------------")
    print( "Generate prediction-task news (textual event sequences)...")
    """
    news_matrix
    """
    news_matrix = np.load(file=args.news_path, allow_pickle=True)
    sents = news_matrix[:,:,0]
    labels = news_matrix[:,:,1]
    sents[np.where(sents == None)] = ''
    max_value = 0
    for i in range(sents.shape[0] * sents.shape[1]):
        max_value = max(max_value, len(sents.reshape(-1)[i]))    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_bert_path)
    print(tokenizer)
    tokenizer.batch_encode_plus([
        'hide new secretions from the parental units | contains no wit , only labored gags,sdadsadsadsadsadsadasdasdasdsadsadsadasdasdasdasdasdasdsadsadas',
        '|'
    ])
    
    daily_news_list = []
    daily_news_input_ids_list = []
    daily_news_token_type_ids_list = []
    daily_news_attention_mask_list = []

    for i in range(news_matrix.shape[0]): 
        one_day_news = tokenizer.batch_encode_plus(
                            batch_text_or_text_pairs = sents[i].tolist(), 
                            truncation=True, 
                            max_length=70,
                            padding='max_length', 
                            return_tensors='pt') 
            
        daily_news_list.append(one_day_news)
        daily_news_input_ids_list.append(one_day_news['input_ids'])
        daily_news_token_type_ids_list.append(one_day_news['token_type_ids'])
        daily_news_attention_mask_list.append(one_day_news['attention_mask'])
    daily_news_input_ids = torch.stack(daily_news_input_ids_list)
    daily_news_token_type_ids = torch.stack(daily_news_token_type_ids_list)
    daily_news_attention_mask = torch.stack(daily_news_attention_mask_list)
    mask_value = torch.cat([torch.tensor([101,102,]), torch.full((68,),0)]).repeat(730,40,1)
    mask = torch.eq(daily_news_input_ids, mask_value)
    daily_valid_news_mask = torch.any(~mask, dim = -1)
    daily_valid_news_mask = daily_valid_news_mask.unsqueeze(-1)
    
    daily_news_input_ids_feature,_ =  generate_dataset(daily_news_input_ids, args,'None') 
    daily_news_token_type_ids_feature,_  =  generate_dataset(daily_news_token_type_ids, args,'None') 
    daily_news_attention_mask_feature,_  =  generate_dataset(daily_news_attention_mask, args,'None') 
    daily_valid_news_mask_feature,_ = generate_dataset(daily_valid_news_mask, args,'None') 

    return ds_feature, ds_target, outbreak_feature, risk_feature, news_matrix_fine_tuning, daily_news_input_ids_feature, daily_news_token_type_ids_feature, daily_news_attention_mask_feature, daily_valid_news_mask_feature, abnormal_news_num_feature, festival_feature, festival_target

class Dataset(torch.utils.data.Dataset): 

    def __init__(self, indices, split, args):
        
        train_len = int(len(indices) * args.train_ratio)
        val_len =  int(len(indices) * args.val_ratio)
        test_len = len(indices) - train_len - val_len
        train_indices, valid_indices, test_indices = torch.utils.data.random_split(dataset = indices, lengths = [train_len, val_len, test_len], generator = torch.Generator().manual_seed(args.seed))
        
        if split == 'train':
            indices = train_indices
        elif split == 'valid':
            indices = valid_indices
        elif split == 'test':
            indices = test_indices
        else: 
            raise ValueError
        
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i): 
        index = self.indices[i]
        return index
    
# loss function
import torch.nn.functional as F
def l1_loss(input_data, target_data, **kwargs):
    """unmasked mae."""
    return F.l1_loss(input_data, target_data)
def l2_loss(input_data, target_data, **kwargs):
    """unmasked mse"""
    return F.mse_loss(input_data, target_data)

# metrics
def mse(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """mean squared error.
    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: mean squared error
    """
    loss = (preds-labels)**2
    return torch.mean(loss)

def rmse(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """root mean squared error.
    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: root mean squared error
    """
    return torch.sqrt(mse(preds=preds, labels=labels))

def mae(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = torch.abs(preds-labels)
    return torch.mean(loss)

def mape(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Masked mean absolute percentage error.
    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    loss = torch.abs(torch.abs(preds-labels)/labels)
    return torch.mean(loss)

def smape(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    This is smape_adjusted, multiple 2 is the smape
    """
    loss = torch.abs(torch.abs(preds-labels)/((torch.abs(preds) + torch.abs(labels))/2))
    
    return torch.mean(loss) 

def masked_MSE(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.eq(true, mask_value)
        pred = torch.masked_select(pred, ~mask)
        true = torch.masked_select(true, ~mask)
    return torch.mean((pred - true) ** 2)

def masked_RMSE(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.eq(true, mask_value)
        pred = torch.masked_select(pred, ~mask)
        true = torch.masked_select(true, ~mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def masked_MAE(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.eq(true, mask_value)
        pred = torch.masked_select(pred, ~mask)
        true = torch.masked_select(true, ~mask)
    return torch.mean(torch.abs(true-pred))

def masked_MAPE(pred, true, mask_value=0):
    if mask_value != None:
        mask = torch.eq(true, mask_value)
        pred = torch.masked_select(pred, ~mask)
        true = torch.masked_select(true, ~mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def masked_sMAPE(pred, true, mask_value=0):
    """
    This is masked_smape_adjusted, multiple 2 is the masked_smape
    """
    if mask_value != None:
        mask = torch.eq(true, mask_value)
        pred = torch.masked_select(pred, ~mask)
        true = torch.masked_select(true, ~mask)
    return torch.mean(torch.div(torch.abs(true - pred), (torch.abs(true) + torch.abs(pred))/2))