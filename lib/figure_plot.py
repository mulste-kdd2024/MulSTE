import matplotlib.pyplot as plt

def train_loss_plot(args, current_time, epoch_train_loss_d, epoch_train_loss_s, epoch_train_loss):

    fig = plt.figure(figsize=(8, 8))
    plt.grid(axis="y",linestyle='-.')
    plt.xticks(fontsize=20)
    plt.title('Train Loss',fontsize=30) 

    ax1 = fig.add_subplot(111)
    ax1.plot(epoch_train_loss_d, label='demand loss', lw=2) 
    ax1.plot(epoch_train_loss_s, label='supply loss', lw=2) 
    ax1.plot(epoch_train_loss, label='total loss', lw=2) 

    ax1.legend(loc=1,fontsize=15) 
    ax1.set_xlabel('epoch', fontsize=20)
    ax1.set_ylabel('loss value', fontsize=20) 
    ax1.tick_params(top=False,bottom=True,left=True,right=False)
    plt.yticks(fontsize=20) 

    plt.show()
    fig.savefig(args.log_dir + '/train_loss_figure_{}.png'.format(current_time))
    
def valid_loss_plot(args, current_time, epoch_valid_loss_d, epoch_valid_loss_s, epoch_valid_loss):
    
    fig = plt.figure(figsize=(8, 8))
    plt.grid(axis="y",linestyle='-.')
    plt.xticks(fontsize=20)
    plt.title('Valid Loss',fontsize=30) 

    ax1 = fig.add_subplot(111)
    ax1.plot(epoch_valid_loss_d, label='demand loss', lw=2) 
    ax1.plot(epoch_valid_loss_s, label='supply loss', lw=2) 
    ax1.plot(epoch_valid_loss, label='total loss', lw=2) 

    ax1.legend(loc=1,fontsize=15) 
    ax1.set_xlabel('epoch', fontsize=20) 
    ax1.set_ylabel('loss value', fontsize=20) 
    ax1.tick_params(top=False,bottom=True,left=True,right=False) 
    plt.yticks(fontsize=20) 
    
    plt.show()
    fig.savefig(args.log_dir + '/valid_loss_figure_{}.png'.format(current_time))

def pred_plot(args, current_time, test_timestamp, outputs_d, targets_d, outputs_s, targets_s):
    
    # delivery station ID
    site_i = 0
    future_i = 0

    fig = plt.figure(figsize=(20, 8))
    plt.grid(axis="y",linestyle='-.')
    plt.xticks(fontsize=20)
    plt.title('Site_{} Demand Prediction Visualization'.format(site_i),fontsize=30) 

    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, 10000])
    ax1.plot(test_timestamp, outputs_d[:,future_i,site_i], label='Demand Pred', lw=2) 
    ax1.plot(test_timestamp, targets_d[:,future_i,site_i], label='Demand Truth', lw=2) 
    
    ax1.legend(loc=1,fontsize=15) 
    ax1.set_xlabel('Sample', fontsize=20) 
    ax1.set_ylabel('Demand Volume', fontsize=20) 
    ax1.tick_params(top=False,bottom=True,left=True,right=False) 
    plt.yticks(fontsize=20) 
    
    plt.show()
    fig.savefig(args.log_dir + '/prediction_visualization_d_{}.png'.format(current_time))
    
    ###########################################################################################################
    
    fig = plt.figure(figsize=(20, 8))
    plt.grid(axis="y",linestyle='-.')
    plt.xticks(fontsize=20)
    plt.title('Site_{} Supply Prediction Visualization'.format(site_i),fontsize=30) 
    
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, 10000])
    ax1.plot(test_timestamp, outputs_s[:,future_i,site_i], label='Supply Pred', lw=2) 
    ax1.plot(test_timestamp, targets_s[:,future_i,site_i], label='Supply Truth', lw=2) 
    
    ax1.legend(loc=1,fontsize=15) 
    ax1.set_xlabel('Sample', fontsize=20) 
    ax1.set_ylabel('Supply Volume', fontsize=20) 
    ax1.tick_params(top=False,bottom=True,left=True,right=False) 
    plt.yticks(fontsize=20) 

    plt.show()
    fig.savefig(args.log_dir + '/prediction_visualization_s_{}.png'.format(current_time))