## Dependency
```bash
pip install -r requirements.txt
```

## Data
Make a directory named `data` in the current directory. Download all the data from this [link](https://drive.google.com/drive/folders/1SFHOyiyxMxpX4lieFebqzuolZ7a73_-L?usp=sharing) and put them under the directory `data`.

## Differential Privacy Experiments
You can try sst2, mnli, qqp, qnli for the `task` argument.

```
python3 main.py \
--gpu 0 \
--n-workers 6 \
--n-clients 6 \
--frac 0.5 \
--lr 0.5 \
--lora-alpha 8 \
--model ffa\
--task "{task}"\
--dist non_iid\
--trial "fed-svd"\
--peft lora\
--n-rnds 100 \
--alpha 0.5 \
--seed 42 \
--lora-rank 8 \
--batch_size 128 \
--backbone roberta-large \
--scheduler-type constant \
--accumulation-steps 1 \
--report-period 10 \
--recalculate-svd-period 1\
--dropout 0.05 \
--n-iter 10 \
--recalculate-svd-period 1 \
--dp \
--eps 6 \
--max-grad-norm 2
```

## Non-private experiments
You can try sst2, mnli, qqp, qnli for the `task` argument.

```
python3 main.py \
--gpu 0 \
--n-workers 6 \
--n-clients 6 \
--frac 0.5 \
--lr 0.5 \
--lora-alpha 8 \
--model ffa\
--task "{task}"\
--dist non_iid\
--trial "fed-svd"\
--peft lora\
--n-rnds 100 \
--alpha 0.5 \
--seed 42 \
--lora-rank 8 \
--batch_size 128 \
--backbone roberta-large \
--scheduler-type constant \
--accumulation-steps 1 \
--report-period 10 \
--recalculate-svd-period 1\
--dropout 0.05 \
--n-iter 10 \
--recalculate-svd-period 1
```
