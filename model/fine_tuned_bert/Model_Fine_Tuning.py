import torch
from transformers import BertModel

class Model_Fine_Tuning(torch.nn.Module):

    def __init__(self, pretrained_bert_path):
        super().__init__() 
        
        self.pretrained = BertModel.from_pretrained(pretrained_bert_path) 

        self.fc = torch.nn.Sequential(torch.nn.Linear(768, 768),
                                      torch.nn.ReLU(), 
                                      torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(768, 2),
                                      torch.nn.Softmax(dim=1))        
        
        self.criterion = torch.nn.CrossEntropyLoss() 

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.pretrained(input_ids=input_ids, 
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        
        # last_hidden_state: (batch_size, seqence_length, hidden_size) -> cls_last_hidden_state: (batch_size, hidden_size)
        cls_last_hidden_state = output.last_hidden_state[:, 0, :]
        
        # logits: (batch_size, num_labels)
        logits = self.fc(cls_last_hidden_state)
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {'loss': loss, 'logits': logits} 