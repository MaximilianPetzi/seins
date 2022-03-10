#Nsims=5
#rm error_h/*
#jljjlj
set -m                  #enabling forcefully job control (necessary?)
for i in {1..5}     #just do 5 at once instead of 10
do
sleep 10
python3 jiggling.py $i 8&
done
#wait for simulations to be full
