<!--
 * @函数说明: 
 * @Author: hongqing
 * @Date: 2021-07-21 11:29:10
 * @LastEditTime: 2021-08-12 15:35:34
-->
# PyTorch implementation of Reinforcement Learning A3C,DQN,DPPO
Use Reinforcement learning to play pazudora!

用强化学习的方式游玩智龙迷城

強化学習を使ってパズドラを遊びます

![pad](data/result/pad.png)
A3C Paper: Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf),A3C refer to morvan's code ["这里"](https://github.com/MorvanZhou/pytorch-A3C)

A3C实现参考了论文和莫凡的代码 在上面

A3Cの実現は論文とmovanさんのコードを参考しました

```
├── README.md
├── data            dataset
├── env             pad environment
├── padA3C            
├── padDQN              
├── testfile
├── main.py         demo to run pytorch --> padA3C/DQN/test
```


DQN未整理，DPPO未完成



## step1.environment
please use [pip install -r requirement.txt] in your command line

第一步使用 pip install -r requirement.txt 安装所需依赖包

pip install -r requirement.txtを入力して環境パッケージをインストールします

## step2.test
python main.py --method test

## step3.train

run 5X6 with 6colors pazudora(4 processes):

python main.py --method A3C --row_size 5 --col_size 6 --color_size 6 --num-processes 4 fps=-1

(if you want see the animation, set fps>0)

跑这个代码就行了 详细参数看代码

実行すればわかる。。。。はず

## result
The results will be saved in the /logs in the root directory. After installing tensorflow and tenforboard, execute tensorboard -- logdir logs to see the results

结果会被保存在根目录的logs文件下，安装TensorFlow 和tenforBoard 后执行 tensorboard --logdir logs

tensorboardを利用して可視化できる、TensorFlow 和tenforBoardをインストールしたあと　tensorboard --logdir logsを実行すると結果が出てくる



## Dependencies

* pytorch >= 0.4.0
* numpy
* gym
* matplotlib
* tensorflow
* tensorBoard
