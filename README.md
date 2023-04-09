
# The origin story of LLMs

All the juice begins with BERT. If we learn BERT, it is fairly easy to retrace
the public models, including [Github GPT3](https://github.com/openai/gpt-3).
GPT3 is an enlargement of BERT. GPT-4 uses RLHF based upon about 20k hours
of RLHF scored conversations by humans (according to rumor).

## purpose unclear

I am not sure how these relate to the goal of running homomorphic encryption. I
did write them down, so perhaps they are relevant in some way. Using the squad_v2
will be an important step in verifying that a fine-tuned model that is tuned with
homomorphic encryption works properly.

This might have a clearer purpose: [Runhouse Github](https://github.com/run-house/runhouse)

https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France
https://huggingface.co/spaces/evaluate-metric/squad_v2
https://huggingface.co/deepset/minilm-uncased-squad2

## Run-SQAUD using pytorch

Fune-tune BERT using the SQUAD scoring method. This usees the latest pytorch from
the legacy community submissions and is an improvement over the original BERT release
because it functions, and its more performant.

The startup code is [run-squad](run.sh) based upon:
https://huggingface.co/docs/transformers/training
This code is more or less [copy pasta](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering) however, I do understand some of the behaviors after attempting to get the original BERT release to fine tune. I am glad I found
 an easier answer, however, the references in original bert release are
**invaluable**.


## Original BERT release

This is hard to get working. I am not sure I was able to fine-tune a model. The
homomorphic algorithm is implemented against the BERT tiny model. Re-implementing
the paper should offer us some insight on the performance impacts in terms of
quality, compute cost, and scdalability with tools like
[ColossalAI](https://github.com/hpcaitech/ColossalAI). I did find, however, I was
able to fine-tune with the community run-squad implementation, so that may
be a better starting point.

https://github.com/maknotavailable/pytorch-pretrained-BERT/
https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6

# Prereqs

- Ensure git has LFS setup properly
```bash
sudo apt install git-lfs
cd $REPO
git-lfs init
```

```
- Install [tensorflow sans Conda](https://www.tensorflow.org/install/pip#linux_setup)
```bash
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
#echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' > $HOME/env.artificialwisdom
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/targets/x86_64-linux/lib:/lib/usr/local/lib/python3.9/dist-packages/nvidia/cudnn/lib' >> $HOME/env.artificialwisdom
$CUDNN_PATH/lib
source  $HOME/env.artificialwisdom
# Verify the install as per the Tensorflow installation
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
- Install [TensorRT sans Conda](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_8)
- The dependencies are difficult to reason about. We **need** a requirements.txt...
