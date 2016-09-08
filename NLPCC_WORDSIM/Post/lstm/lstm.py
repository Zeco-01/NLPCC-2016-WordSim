# encoding=UTF-8


from __future__ import print_function

import sys
import time
from collections import OrderedDict
import numpy
import six.moves.cPickle as pickle
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt
import prepare_input
from Com import macro
from Eval import eval
from Com import utils
import seaborn as sns
from pandas import DataFrame
import codecs
from Post import merge


theano.config.floatX = 'float32'
config.floatX = 'float32'
config.exception_verbosity = 'high'
# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)  # 返回分割之后的idx_list


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5).astype(config.floatX)
    return proj.astype(config.floatX)


def _p(pp, name):  # 连接字符串
    return '%s_%s' % (pp, name)


def load_trained_embeddings(filename):
    w2v_model = prepare_input.get_w2v_model(filename)
    wemb = numpy.zeros((0, 600)).astype(config.floatX)
    dictionary, ids = prepare_input.get_all_dictionary()
    zeros = [0] * 300

    for id, token in dictionary.id2token.items():
        try:

            temp = w2v_model[token].tolist()
            temp.extend(zeros)
            emb = numpy.array(temp)
            emb = emb.astype(config.floatX)
            wemb = numpy.row_stack([wemb, emb]).astype(config.floatX)
            # wemb = numpy.row_stack([wemb, (w2v_model[token]).astype(config.floatX)])
        except KeyError:
            randn = numpy.random.rand(1, 300)
            rand = (0.01 * randn)
            temp = rand[0].tolist()
            temp.extend(zeros)
            emb = numpy.array(temp).astype(config.floatX)
            wemb = numpy.row_stack([wemb, emb]).astype(config.floatX)

    return wemb.astype(config.floatX)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    """
    params = OrderedDict()
    # embedding

    params['Wemb'] = load_trained_embeddings(macro.MODELS_DIR + '/fml_org_bdnews_xieso_w2v.bin')

    # randn = numpy.random.rand(options['n_words'],
    #                           options['dim_proj'])  # 10000单词，128维度，随机初始化
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)  # 为啥除100？

    params = param_init_lstm(options,
                             params,
                             prefix='lstm')
    # classifier
    # U 和 b 是最后逻辑回归的参数
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)  # 随机生成U
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)  # b初始为0

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def ortho_weight(ndim):  # 随机生成方阵 奇异值分解
    W = numpy.random.randn(ndim, ndim).astype(config.floatX)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    # 随机初始化lstm_W lstm_U
    # lstm_W 和 lstm_U都是ifoc对应的四个矩阵拼起来的
    # lstm_b初始为0
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1).astype(config.floatX)
    params[_p(prefix, 'W')] = W.astype(config.floatX)
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1).astype(config.floatX)
    params[_p(prefix, 'U')] = U.astype(config.floatX)
    b = numpy.zeros((4 * options['dim_proj'],)).astype(config.floatX)
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):  # 分割
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):  # 一起计算ifoc四个值，返回h和c
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')]).astype(config.floatX)  # h_ * lstm_U
        preact += x_  # x_ 已经计算好的Wx+b 这里再加上preact就行了

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj'])).astype(config.floatX)  # 公式中的it
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj'])).astype(config.floatX)  # 公式中的ft
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj'])).astype(config.floatX)  # 公式中的ot
        c = tensor.tanh(_slice(preact, 3, options['dim_proj'])).astype(config.floatX)  # 公式中的Ct波浪

        c = (f * c_ + i * c).astype(config.floatX)  # 公式中的Ct
        c = (m_[:, None] * c + (1. - m_)[:, None] * c_).astype(config.floatX)  # 不明觉厉
        # 网上看的：对于长度小于maxlen的句子，会补零，但是在这些0位置处，
        # memory cell的状态采用了句子的最后一个单词计算的状态memory cell c进行填充。

        h = (o * tensor.tanh(c)).astype(config.floatX)  # 公式中的ht
        h = (m_[:, None] * h + (1. - m_)[:, None] * h_).astype(config.floatX)  # 不明觉厉

        return h.astype(config.floatX), c.astype(config.floatX)

    # state_below 是 emb传进来的
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')]).astype(config.floatX)  # 并行计算所有的W*x+b

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]  # 即h矩阵 所有的单步h合起来的一个3D矩阵


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost, d1, d2):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y, d1, d2], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared', )

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def transform(dis_vecs, max_len):
    result = numpy.zeros((len(dis_vecs), max_len))
    i = 0
    for dis in dis_vecs:
        row = result[i]
        j = 0
        while j < len(dis):
            row[j] = dis[j]
            j += 1
        i += 1

    try:
        temp = result.astype(config.floatX)
    except:
        pass
    return temp.transpose()


def slice(dis_vecs):
    d1 = []
    d2 = []
    for vec in dis_vecs:
        d1.append(vec[0])
        d2.append(vec[1])
    return d1, d2


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    def extend(emb, dis1, dis2):

        zeros_300 = [0] * 300
        zeros_150 = [0] * 150
        ones_150 = [1] * 150
        zeros_300.extend(ones_150)
        zeros_300.extend(zeros_150)

        mask1 = theano.shared(numpy.array(zeros_300))

        zeros_450 = [0] * 450

        ones_1502 = [1] * 150

        zeros_450.extend(ones_1502)

        mask2 = theano.shared(numpy.array(zeros_450))

        # vec_list1 = tensor.dot(dis1, mask1)
        # vec_list2 = tensor.dot(dis2, mask2)
        vec_list1 = dis1 * mask1
        vec_list2 = dis2 * mask2
        zeros_300 = [0] * 300
        ones_300 = [1] * 300
        ones_300.extend(zeros_300)
        maskx = theano.shared(numpy.array(ones_300).transpose())
        new_emb = tensor.dot(emb, maskx)
        # new_emb = tensor.add(emb, vec_list1)
        # new_emb = tensor.add(new_emb, vec_list2)
        new_emb = new_emb + vec_list1 + vec_list2
        # return new_emb.astype(config.floatX)
        return new_emb

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    d1 = tensor.matrix('d1', dtype=config.floatX)
    d2 = tensor.matrix('d2', dtype=config.floatX)
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    # x:句子是竖着的

    x_flat = x.flatten()
    d1_f = d1.flatten()
    d2_f = d2.flatten()

    emb = tparams['Wemb'][x_flat]  # 在这里把后300维改成距离向量

    new_semb, updates = theano.scan(fn=extend, sequences=[emb, d1_f, d2_f], name='emb_extend',
                                    n_steps=n_samples * n_timesteps)
    new_emb = new_semb.reshape([n_timesteps, n_samples, options['dim_proj']]).astype(config.floatX)
    # 从tparams['Wemb']中取出x.flatten()中的元素对应的行，拼起来
    # 然后reshape

    proj = lstm_layer(tparams, new_emb, options,
                      prefix='lstm',
                      mask=mask)
    # proj:大h矩阵
    # 对3D的h矩阵，各个时序进行Mean Pooling，得到2D矩阵，
    proj = (proj * mask[:, :, None]).sum(axis=0).astype(config.floatX)
    proj = (proj / mask.sum(axis=0)[:, None]).astype(config.floatX)
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b']).astype(
        config.floatX)  # softmax 处理 得到最后的pred值

    f_pred_prob = theano.function([x, mask, d1, d2], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask, d1, d2], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean().astype(config.floatX)  # 根据pred和y计算cost

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, d1, d2


def pred_probs(f_pred_prob, prepare_data, data, iterator, dis_vecs, verbose=True):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 11)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y, vecs = prepare_data([data[0][t] for t in valid_index],
                                        numpy.array(data[1])[valid_index],
                                        maxlen=None, dis_vecs=[dis_vecs[t] for t in valid_index])
        d1, d2 = slice(vecs)
        d1 = transform(d1, x.shape[0])
        d2 = transform(d2, x.shape[0])
        pred_probs = f_pred_prob(x, mask, d1, d2)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, dis_vecs, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y, vecs = prepare_data([data[0][t] for t in valid_index],
                                        numpy.array(data[1])[valid_index],
                                        maxlen=None, dis_vecs=[dis_vecs[t] for t in valid_index])
        d1, d2 = slice(vecs)
        d1 = transform(d1, x.shape[0])
        d2 = transform(d2, x.shape[0])
        preds = f_pred(x, mask, d1, d2)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def probs_2_score(probs, ids):
    score = []
    id_dis = []
    for row in probs:
        i = 1
        sum = 0
        for p in row:
            sum += i * p
            i += 1
        score.append(sum)
    j = 1
    id_temp = ids[0]
    id_dis.append(ids[0])
    scores = []
    sum_temp = score[0]
    count = 1
    while j < len(score):
        if ids[j] == id_temp:
            count += 1
            sum_temp += score[j]
        else:
            avg = (sum_temp * 1.0) / count
            count = 1
            sum_temp = score[j]
            scores.append(avg)
            id_dis.append(ids[j])
            id_temp = ids[j]
        j += 1
    avg = (sum_temp * 1.0) / count
    scores.append(avg)
    return scores, id_dis


def train_lstm(
        dim_proj=600,  # word embedding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=adadelta,
        # sgd, adadelta and rmsprop available, sgd very hard to use,
        # not recommended (probably need momentum and decaying learning rate).
        saveto='lstm_model.npz',  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=500,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.

        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
        # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
        part=1
):
    # Model options
    model_options = locals().copy()
    print("model options", model_options)
    load_data = prepare_input.load_data
    prepare_data = prepare_input.prepare_data
    print('Loading data')
    train, valid, test, dis_vecs_train, ids_train, dis_vecs_valid, ids_valid, dis_vecs_test, ids_test = load_data(
        n_words=n_words, valid_portion=0.05,
        maxlen=maxlen, part=part)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
        dis_vecs_test = [dis_vecs_test[n] for n in idx]
        ids_test = [ids_test[n] for n in idx]
    ydim = numpy.max(train[1])

    model_options['ydim'] = ydim + 1

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, d1, d2) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y, d1, d2], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))  # 计算梯度
    f_grad = theano.function([x, mask, y, d1, d2], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost, d1, d2)  # 梯度更新算法

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                dis = [dis_vecs_train[t] for t in train_index]
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y, dis_vecs = prepare_data(x, y, dis_vecs=dis)  # x句子是竖着的
                ds1, ds2 = slice(dis_vecs)
                ds1 = transform(ds1, x.shape[0])
                ds2 = transform(ds2, x.shape[0])
                n_samples += x.shape[1]
                cost = f_grad_shared(x, mask, y, ds1, ds2)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:  # 存储
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto + str(part), history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % (saveto + str(part)), 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:  # 验证
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf, dis_vecs_train)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid, dis_vecs_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test, dis_vecs_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                                valid_err <= numpy.array(history_errs)[:,
                                             0].min()):
                        best_p = unzip(tparams)  # 记录最好参数
                        bad_counter = 0

                    print(('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err))

                    if (len(history_errs) > patience and
                                valid_err >= numpy.array(history_errs)[:-patience,
                                             0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, dis_vecs=dis_vecs_train)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid, dis_vecs=dis_vecs_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test, dis_vecs=dis_vecs_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    if saveto:
        numpy.savez(saveto + str(part), train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


def init_params_empty():
    params = OrderedDict()
    params['Wemb'] = ''
    params['U'] = ''
    params['b'] = ''
    params[_p('lstm', 'W')] = ''

    params[_p('lstm', 'U')] = ''

    params[_p('lstm', 'b')] = ''
    return params


def combine(idl, ids_dis, score_all, scores):
    id_score1 = zip(idl, score_all)
    id_score2 = zip(ids_dis, scores)
    d1 = OrderedDict((name, value) for name, value in id_score1)
    d2 = OrderedDict((name, value) for name, value in id_score2)
    for id in idl:
        if id in ids_dis:
            d1[id] = max([d1[id], d2[id]])

    return d1.values()


def merge2max(s1, s2):
    result = []
    for ss1, ss2 in zip(s1, s2):
        result.append(max([ss1, ss2]))
    return result


def compare(score_1, score_2):
    for s1, s2 in zip(score_1, score_2):
        if s1 != s2:
            print(s1, '---', s2)


def test_lstm(
        dim_proj=600,  # word embedding dimension and LSTM number of hidden units.
        n_words=100000,  # Vocabulary size
        # sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentum and decaying learning rate).
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        # Parameter for extra option
        noise_std=0.,
        # This frequently need a bigger model.
        test_size=-1,  # If >0, we keep only this number of test example.
        use_dropout=True,  # if False slightly faster, but worst test error
        part=1
):
    start_time = time.time()
    idl, w1l, w2l, score_v, headline = utils.read2wordlist([(macro.RESULTS_DIR, 'fml_google_en_w2v.result')])
    idl, w1l, w2l, score_goldern, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])

    # Model options
    model_options = locals().copy()
    print("model options", model_options)
    load_data = prepare_input.load_data
    prepare_data = prepare_input.prepare_data
    print('Loading data')
    train, valid, test, dis_vecs_train, ids_train, dis_vecs_valid, ids_valid, dis_vecs_test, ids_test = load_data(
        n_words=n_words, valid_portion=0.05,
        maxlen=maxlen, part=part)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
        dis_vecs_test = [dis_vecs_test[n] for n in idx]
        ids_test = [ids_test[n] for n in idx]
    ydim = numpy.max(train[1])

    model_options['ydim'] = ydim + 1
    print('Loading model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params_empty()

    load_params('lstm_model.npz' + str(part) + '.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, d1, d2) = build_model(tparams, model_options)
    it = get_minibatches_idx(len(test[0]), valid_batch_size)
    probs = pred_probs(f_pred_prob, prepare_data, test, it, dis_vecs=dis_vecs_test)
    scores, id_dis = probs_2_score(probs, ids_test)
    new_score = combine(idl, id_dis, score_v, scores)
    out_file = codecs.open(macro.RESULTS_DIR + '/lstm_w2v' + str(part) + '.txt', 'w', 'utf-8')
    out_file.write('ID\tWord1\tWord2\tScore\t\r\n')
    for id, word1, word2, score in zip(idl, w1l, w2l, new_score):
        line = id + '\t' + word1 + '\t' + word2 + '\t' + str(score) + '\r\n'
        out_file.write(line)
    out_file.close()

    # print(eval.spearmanr(score_v, score_goldern)[0])
    # print(eval.spearman(new_score, score_goldern)[0])

    idl, w1l, w2l, score_old, headline = utils.read2wordlist([(macro.RESULTS_DIR, 'best_without_lstm.txt')])
    f_c = macro.RESULTS_DIR + '/evatestdata3_goldern500_cilin.txt'
    f_v = macro.RESULTS_DIR + '/lstm_w2v' + str(part) + '.txt'
    last_score = merge.merge_2_list(f_v, f_c, macro.MAX)
    temp = eval.spearman(last_score, score_goldern)[0]
    print(eval.spearman(score_old, score_goldern)[0])
    print(temp)

    dataset = {
        'pred': last_score,
        'goldern': score_goldern
    }
    frame = DataFrame(dataset)
    sns.jointplot('goldern', 'pred', frame, kind='reg', stat_func=eval.spearmanr)

    plt.xlim([1, 10])
    plt.ylim([1, 10])
    plt.savefig('%s/%s.png' % (macro.PICS_DIR, ('lstm' + str(part))))
    end_time = time.time()
    print(('Testing took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return last_score


def test_2(part):
    pp = numpy.load('lstm_model.npz' + str(part) + '.npz')

    for kk, vv in pp.items():
        if kk == 'Wemb':
            return vv


if __name__ == '__main__':
    # for i in range(1, 6):
    #     train_lstm(
    #         dim_proj=600,
    #         n_words=100000,
    #         max_epochs=100,
    #         test_size=-1,
    #         part=i
    #     )

    # print(test_2(3))


    last_scores = []
    max_score = []

    for i in range(1, 6):
        last_scores.append(test_lstm(part=i))
    idl, w1l, w2l, score_goldern, headline = utils.read2wordlist([(macro.CORPUS_DIR, '500_2.csv')])
    temp = last_scores[0]
    for s in last_scores[1:]:
        max_score = merge2max(temp, s)
        temp = max_score
    print('max_score: ', eval.spearman(max_score, score_goldern), eval.pearson(max_score, score_goldern))
