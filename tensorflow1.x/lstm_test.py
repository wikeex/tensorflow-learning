import numpy as np
import tensorflow as tf
import reader
DATA_PATH = 'E:/datasets/ptb'
VOCAB_SIZE = 10000

HIDDEN_SIZE = 200  # lstm隐含层神经元数
NUM_LAYERS = 2  # lstm结构层数
LEARNING_RATE = 1.0  # 学习率
KEEP_PROB = 0.5  # Dropout保留率
MAX_GRAD_NORM = 5  # 控制梯度膨胀的系数

TRAIN_BATCH_SIZE = 20  # 训练batch尺寸
TRAIN_NUM_STEP = 35  # 训练数据的截断长度

EVAL_BATCH_SIZE = 1  # 测试batch尺寸
EVAL_NUM_STEP = 1  # 测试数据的截断长度
NUM_EPOCH = 2  # 训练轮数


class PTBModel:
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size  # 语料库分集
        self.num_steps = num_steps  # 时间步

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # 输入数据placeholder
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # 输出数据placeholder

        cells = []
        for _ in range(NUM_LAYERS):
            # 基本lstm单元，隐含状态数和输出特征维度都为HIDDEN_SIZE
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE)
            if is_training:
                # 每个lstm单元外包裹一个DropoutWrapper
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)

            cells.append(lstm_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)  # 构建多层rnn网络结构

        self.initial_state = cell.zero_state(batch_size, tf.float32)  # 初始化网络参数为0
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])  # 创建词嵌入变量

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)  # 单词索引转化为词向量

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)  # 训练时执行dropout操作
        # 定义输出层
        outputs = []  # 定义lstm输出列表
        state = self.initial_state  # 保存不同batch中lstm的状态，初始化为0
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()  # 对下面的变量进行复用
                cell_output, state = cell(inputs[:, time_step, :], state)  # 输入数据开始训练，state为历史信息
                outputs.append(cell_output)  # 每个时间步的输出添加到列表中
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])  # 将输出列表拼接成张量
        # 定义softmax层
        softmax_weight = tf.get_variable('softmax_w', [HIDDEN_SIZE, VOCAB_SIZE])  #
        softmax_bias = tf.get_variable('softmax_b', [VOCAB_SIZE])

        logits = tf.matmul(output, softmax_weight) + softmax_bias
        # 定义损失函数
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],  # 表示预测分类的置信度
            [tf.reshape(self.targets, [-1])],  # 表示预期目标为one-hot类型
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]  # 各类损失计算权重均为1
        )
        self.cost = tf.reduce_sum(loss) / batch_size  # 求得每batch的损失
        self.final_state = state  # 更新整个lstm网络状态

        if not is_training:
            return
        trainable_variables = tf.trainable_variables()  # 得到所有trainable=True的变量

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)  # 梯度裁剪
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)  # 梯度下降优化器
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))  # 优化操作应用到变量上


def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0  # 整体代价
    iters = 0  # 迭代次数
    state = session.run(model.initial_state)  # 初始化模型状态

    for step in range(epoch_size):
        x, y = session.run(data)  # 将训练数据拆分成训练部分和标签部分
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y, model.initial_state: state}
        )  # 开始训练
        total_costs += cost  # 整体代价
        iters += model.num_steps  #

        if output_log and step % 100 == 0:
            with open('lstm_run_recode.txt', 'a') as f:
                f.write('After %d steps, perplexity is %.3f\n' % (step, np.exp(total_costs / iters)))
            print('After %d steps, perplexity is %.3f' % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)  # 计算混乱度


def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)  # 读取数据集中的数据

    train_data_len = len(train_data)  # 计算数据长度
    train_batch_len = train_data_len  # 计算batch长度
    train_epoch_size = (train_batch_len - 1)  # 计算该epoch训练次数

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len
    valid_epoch_size = (valid_batch_len - 1)

    test_data_len = test_batch_len = len(test_data)
    test_epoch_size = (test_batch_len - 1)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)  # 随机数初始化
    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)  # 实例化训练模型

    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)  # 实例化评估模型

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)  # 生成训练数据序列
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)  # 生成评估数据序列
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)  # 生成测试数据序列

        coord = tf.train.Coordinator()  # 管理多线程的协调器
        threads = tf.train.start_queue_runners(sess=session, coord=coord)  # 启动多线程

        for i in range(NUM_EPOCH):
            with open('lstm_run_recode.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('In iteration: %d' % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)  # 训练模型

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)  # 评估
            with open('lstm_run_recode.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('Epoch: %d Validation Perplexity: %.3f' % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)  # 测试
        with open('lstm_run_recode.txt', 'a') as f:
            f.write('In iteration: %d\n' % (i + 1))
        print('Test Perplexity: %.3f' % test_perplexity)

        coord.request_stop()  # 请求停止多线程
        coord.join(threads)  # 直到所有线程结束


if __name__ == '__main__':
    main()

