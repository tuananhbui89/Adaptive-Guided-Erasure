import os 
import subprocess

def rsync_to_remote(args):
    LOCAL_TO_TRUNG = "rsync -zarv --include='*/' --include='*.py' --include='*.sh' --include='*.csv' --include='*.md' --include='*.yaml' --exclude='*' --exclude='*__pycache__' /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ tony@130.194.131.137:/mnt/SSD1/tony/GenerativeAI/Forget-Me-Not/"
    LOCAL_TO_TVUONG = "rsync -zarv --include='*/' --include='*.py' --include='*.sh' --include='*.csv' --include='*.md' --include='*.yaml' --exclude='*' --exclude='*__pycache__' /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ tvuong@m3.massive.org.au:/home/tvuong/pb90_scratch/tvuong/bta/Forget-Me-Not/"
    LOCAL_TO_BTA = "rsync -zarv --include='*/' --include='*.py' --include='*.sh' --include='*.csv' --include='*.md' --include='*.yaml' --exclude='*' --exclude='*__pycache__' /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ tbui@m3.massive.org.au:/home/tbui/pb90/bta/workspace/GenerativeAI/Forget-Me-Not/"
    LOCAL_TO_FITCLUSTER = "rsync -zarv --include='*/' --include='*.py' --include='*.sh' --include='*.csv' --include='*.md' --include='*.yaml' --exclude='*' --exclude='*__pycache__' /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ bta@fitcluster02.mpc.monash.edu:/home/bta/Workspace/GenerativeAI/Forget-Me-Not/"

    TRUNG_TO_LOCAL = "rsync -zarvh --progress tony@130.194.131.137:/mnt/SSD1/tony/GenerativeAI/Forget-Me-Not/evaluation_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ --exclude='*.pt'" +\
                    " && rsync -zarvh --progress tony@130.194.131.137:/mnt/SSD1/tony/GenerativeAI/Forget-Me-Not/data  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/" +\
                    " && rsync -zarvh --progress tony@130.194.131.137:/mnt/SSD1/tony/GenerativeAI/Forget-Me-Not/*.log  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/"
    TVUONG_TO_LOCAL = "rsync -zarvh --progress tvuong@m3.massive.org.au:/home/tvuong/pb90_scratch/tvuong/bta/Forget-Me-Not/evaluation_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ --exclude='*.pt'" +\
                    " && rsync -zarvh --progress tvuong@m3.massive.org.au:/home/tvuong/pb90_scratch/tvuong/bta/Forget-Me-Not/invest_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/"
    BTA_TO_LOCAL = "rsync -zarvh --progress tbui@m3.massive.org.au:/home/tbui/pb90/bta/workspace/GenerativeAI/Forget-Me-Not/evaluation_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ --exclude='*.pt'" +\
                    " && rsync -zarvh --progress tbui@m3.massive.org.au:/home/tbui/pb90/bta/workspace/GenerativeAI/Forget-Me-Not/invest_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/"
    FITCLUSTER_TO_LOCAL = "rsync -zarvh --progress bta@fitcluster02.mpc.monash.edu:/home/bta/Workspace/GenerativeAI/Forget-Me-Not/evaluation_folder  /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/ --exclude='*.pt'" +\
                    " && rsync -zarvh --progress bta@fitcluster02.mpc.monash.edu:/home/bta/Workspace/GenerativeAI/Forget-Me-Not/invest_folder /Users/tbui0020/Workspace/GenerativeAI/Forget-Me-Not/"

    TVUONG_MODELS_TO_TRUNG = "rsync -zarvh --progress tvuong@m3.massive.org.au:/home/tvuong/pb90_scratch/tvuong/bta/Forget-Me-Not/models /mnt/SSD1/tony/GenerativeAI/Forget-Me-Not/"

    BTA_MODELS_TO_TRUNG = "rsync -zarvh --progress tbui@m3.massive.org.au:/home/tbui/pb90_scratch/bta/workspace/datasets/ /mnt/SSD1/tony/GenerativeAI/datasets/"


    TVUONG_INTERACTIVE_JOB = "smux new-session --gres=gpu:A100:1 --partition=fit --qos=fitq"

    if args.task == 'local_to_trung':
        cmd = LOCAL_TO_TRUNG
    elif args.task == 'local_to_bta':
        cmd = LOCAL_TO_BTA
    elif args.task == 'local_to_tvuong':
        cmd = LOCAL_TO_TVUONG
    elif args.task == 'local_to_fitcluster':
        cmd = LOCAL_TO_FITCLUSTER
    elif args.task == 'local_to_all':
        cmd = LOCAL_TO_TRUNG + ' && ' + LOCAL_TO_TVUONG + ' && ' + LOCAL_TO_BTA

    elif args.task == 'trung_to_local':
        cmd = TRUNG_TO_LOCAL
    elif args.task == 'tvuong_to_local':
        cmd = TVUONG_TO_LOCAL
    elif args.task == 'bta_to_local':
        cmd = BTA_TO_LOCAL
    elif args.task == 'fitcluster_to_local':
        cmd = FITCLUSTER_TO_LOCAL

    elif args.task == 'tvuong_models_to_trung':
        cmd = TVUONG_MODELS_TO_TRUNG
    elif args.task == 'tvuong_interactive_job':
        cmd = TVUONG_INTERACTIVE_JOB

    elif args.task == 'connect_trung':
        cmd = 'ssh tony@130.194.131.137'
    elif args.task == 'connect_bta':
        cmd = 'ssh tbui@m3-login3.massive.org.au'
    elif args.task == 'connect_tvuong':
        cmd = 'ssh tvuong@m3-login3.massive.org.au'
        
    else:
        raise ValueError('Task not recognized')
    
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to perform')
    args = parser.parse_args()
    rsync_to_remote(args)