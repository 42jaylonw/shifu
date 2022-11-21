PYTHONPATH=.
FILE=logs/nohup.out
source ~/anaconda3/bin/activate rl  # replace to your env name
read -p "Please Enter you command: `echo $'\n> '`" input
nohup $input >> $FILE &
echo $! >> logs/pids.txt
# todo: push run_logs into same dir as task
