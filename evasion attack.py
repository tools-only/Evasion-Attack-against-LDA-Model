import copy
import random
import pickle
import bisect
import lda
import lda.datasets
import collections
import numpy as np
import pandas as pd

def infer(model, x, n_iter=500):
    '''
    infer the new documents 
    
    Parameters
    ----------
    x: array-like, shape(, n_features)
        Testing vector, where n_features is the number of features.
    '''
    global voc
    global n_topics
    rng = np.random.mtrand._rand
    rands = rng.rand(1024**2 // 8)
    n_rand = rands.shape[0]
    alpha = np.repeat(model.alpha, n_topics).astype(np.float64)
    eta = np.repeat(model.eta, len(voc)).astype(np.float64)
    eta_sum = np.sum(eta)
    nzw = model.nzw_ # topic-word matrix
    
    # initialize
    nz = np.zeros(n_topics)
    ZS = np.empty_like(x, dtype=np.intc) # test doc中每个单词的topic组成的list
    dist_sum = np.zeros(n_topics)
    for index, word in enumerate(x):
        w = voc[word]
        z_new = index % n_topics
        ZS[index] = z_new
        nzw[z_new, w] += 1
        nz[z_new] += 1
    
    # sampling
    for it in range(n_iter):
        for index, word in enumerate(x):
            w = voc[word]
            z = ZS[index]
            nzw[z, w] -= 1
            nz[z] -= 1
            dist_cum = 0
            for k in range(n_topics):
                dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (nz[k] + alpha[k])
                dist_sum[k] = dist_cum

            r = rands[index % n_rand] * dist_cum # dist_cum == dist_sum[-1]
            z_new = bisect.bisect_left(dist_sum, r)

            ZS[index] = z_new
            nzw[z_new, w] += 1
            nz[z_new] += 1
            
    return np.array(nz)
        
def pos_filter(x, word):
	'''
	pos_filter: Remove words that are inconsistent with the current part of speech from the candidate set
	'''
	wordq = [['JJ', 'JJR', 'JJS'], ['NN', 'NNS', 'NNP', 'NNPS'], ['PRP', 'PRP$'], ['RB', 'RBR', 'RBS'],
         ['VB', 'VBD', 'VBG', 'VBN', 'VBZ']]

    if x[0][0] == word:
        del x[0]
    pos_list = nltk.pos_tag(np.array(x)[:, 0].tolist())
    pos_list = np.array(pos_list)[:, 1]
    target = nltk.pos_tag([word])
    target_list = []
    for i in wordq:
        if target[0][1] in i:
            target_list = i
    try:
        index = [i for i in range(len(np.array(pos_list))) if (pos_list[i] == j for j in target_list)]
    except:
        return np.array([word, 1.0])
    res = [tuple(i) for i in np.array(x)[index]]
    return res

def calculate_diff(word_count, voc, Nkv, target):
    diff = 0
    for (word, count) in word_count.items():
        wi = voc[word]
        diff += count * (Nkv[:, wi][target]) / sum(Nkv[:, wi])

    return diff

def rank_cal(x, target, result, sim_s, flag):
	'''
	Results statistics
	return: result [rank after attack, target topic id, score after attack, word embedding similarity loss]
	'''
    new_x = x / sum(x)
    output = {}
    for index, k in enumerate(new_x):
        output[index] = k

    output = sorted(output.items(),key = lambda x:x[1],reverse = True)
    for index, k in enumerate(output):
        if k[0] != target and k[1] == 0:
            result[flag].append([index + 1, k[0], k[1], sim_s])
            break
        if k[0] == target:
            result[flag].append([index + 1, k[0], k[1], sim_s])
            break
    return result

def SeakForWord(doc, target, nd, Nkv, length, Attack_algo, alpha = 0.005):
    '''
    Input
        doc: victim submission
        rev_dt: the predicted topic distribution of the target reviewer
        nd: the predicted topic sampling count of the victim submission
        Nkv: the training topic-word sampling count of whole voc
        alpha: modification budget
    Output
        attack path, adversarial topic sampling count estimation, adversarial submission
    '''
    global voc
    global path
    # 获取solution space
    solution_space = []
    for i in sub_sample:
        if i in we['word'].values and i not in solution_space:
            solution_space.append(i)

    new_we = we[we['word'].isin(solution_space)]
    new = copy.deepcopy(new_we)
    iterations = int(length * alpha)
    nd = nd.astype('float64')
    sim_sum = 0
    adv_solu = {} # 记录攻击路径
    already = []
    word_counts = collections.Counter(doc)
    n_u_phi = calculate_diff(word_counts, voc, Nkv, target)
        
    for i in range(iterations):
        # 寻找word candidate(v1.0仅考虑目标topic的值最大化)
        word_cos_max = float('-inf')
        for index, word in enumerate(doc): # 在doc中寻找被替代的词 v
            word_index = voc[word] # 被替换词在词库中的index
            if word not in new['word'].values or index in already:
                continue

            em_word_index = int(list(new[new['word'].isin([word])].index.values)[0]) # 被替换词在embedding词库中的index            
            # get similar words
            ori_phi = Nkv[:, word_index] / sum(Nkv[:, word_index])
            similar_words = new.loc[em_word_index, 'similar_words']

            if similar_words == []:
                continue
            elif len(similar_words) > 10:
                similar_words = similar_words[:10]
            for (sim_word, sim_value) in similar_words: # 在相似词库中寻找替代词 v'
                sim_index = voc[sim_word] # 替换词在词库中的index
                # level 0 
                if Attack_algo == 'Ours_level_0':
                    new_phi = Nkv[:, sim_index] / sum(Nkv[:, sim_index])
                    phi_diff = (new_phi - ori_phi)
                    new_mk = nd + phi_diff # 得到新文章的topic采样分布估计
                    new_n_d_sub = new_mk / sum(new_mk) # 概率
                    ori_n_d_sub = nd / sum(nd)
                    score = new_n_d_sub[target] - ori_n_d_sub[target] 
                else: 
                    phi_v_new = Nkv[:, sim_index] / sum(Nkv[:, sim_index])
                    new_n_u_phi = n_u_phi - ori_phi[target] + phi_v_new[target]
                    # level 1
                    if Attack_algo == 'Ours_level_1':
                        score = phi_v_new[target] * new_n_u_phi - ori_phi[target] * n_u_phi

                    elif Attack_algo == 'Ours_level_2':
                        score = (phi_v_new[target] * new_n_u_phi) ** 2 - (ori_phi[target] * n_u_phi) ** 2

                    elif Attack_algo == 'Ours_level_2_dis':
                        score = (phi_v_new[target] * new_n_u_phi) ** 2 - (ori_phi[target] * n_u_phi) ** 2
                        score /= float(sim_value)

                    elif Attack_algo == 'Ours_level_3':
                        score = (phi_v_new[target] * new_n_u_phi) ** 3 - (ori_phi[target] * n_u_phi) ** 3
                    elif Attack_algo == 'Ours_level_4':
                        score = (phi_v_new[target] * new_n_u_phi) ** 4 - (ori_phi[target] * n_u_phi) ** 4
                    elif Attack_algo == 'Ours_level_5':
                        score = (phi_v_new[target] * new_n_u_phi) ** 5 - (ori_phi[target] * n_u_phi) ** 5
                    elif Attack_algo == 'Ours_level_6':
                        score = (phi_v_new[target] * new_n_u_phi) ** 6 - (ori_phi[target] * n_u_phi) ** 6
            
                if score > word_cos_max:
                    word_cos_max = score
                    word_candidate = (word, sim_word, score, index, float(sim_value))
                elif score == word_cos_max:
                    if word_candidate and word_candidate[-1] > float(sim_value):
                        word_cos_max = score
                        word_candidate = (word, sim_word, score, index, float(sim_value))
                        
        print(Attack_algo, '待修改单词：', word_candidate[0])
        pos_candidate = [ind for ind, x in enumerate(doc) if (x == word_candidate[0]) and (ind not in adv_solu.keys())]
        print(Attack_algo, '待修改位置：', pos_candidate[0])

        pos = pos_candidate[0]
        print(Attack_algo, '选定攻击位置', pos)
        print(Attack_algo, '替换为：', word_candidate[1], ' ', voc[word_candidate[1]])

        doc[pos] = word_candidate[1] # 修改原文
        adv_solu[(pos, word_candidate[0])] = word_candidate[1] # 记录攻击路径
        already.append(word_candidate[3])
        print(word_candidate)
        sim_sum += word_candidate[-1]

        # 更新词频
        word_counts[word_candidate[0]] -= 1
        word_counts[word_candidate[1]] += 1

        # 更新nd
        if Attack_algo == 'Ours_level_0':
            nd += (word_candidate[2])
        else:
            n_u_phi += (phi_v_new[target] - ori_phi[target])
              
            
    return adv_solu, nd, doc, sim_sum

def main():
	n_topics = 120 
	'''
	load the data and lda model
	'''
	# training data
	f = open('D:/论文工作/peer review system/第三篇工作/第三篇工作/code/nips dataset/服务器代码/training_nostem.csv', 'rb')
	dat = pd.read_csv(f)

	# test data
	f_s = open(path_dir + datasets + '/submissions_nostem.csv')
	submissions = pd.read_csv(f_s)
	submissions['paper_text_tokens'] = submissions['paper_text_tokens'].map(lambda x: eval(x))

	# well-trained gibbs lda model
	path_dir = './dataset/'
	datasets = 'nips'
	with open("D:\\论文工作\\peer review system\\第三篇工作\\代码\\dataset\\nips\\gibbs_lda_nostem_5000.pickle",'rb') as file:
	    model = pickle.loads(file.read())

	# load the vocabulary
	path = 'lda_data/'
	# X = lda.datasets.load_reuters(path)
	vocab = lda.datasets.load_reuters_vocab(path)
	from collections import defaultdict
	voc = defaultdict()
	for index, i in enumerate(vocab):
	    word = i.replace('\n', '')
	    if not word:
	        continue
	    voc[word] = index

	# similar words under pos constrains
	new_w = pd.read_csv('D:\论文工作\peer review system\第三篇工作\代码\word embedding\similar_words_100.csv')
	new_w['similar_words'] = new_w['similar_words'].map(lambda x: eval(x))

	for i in new_w.index:
	    new_w.set_value(i, 'similar_words', pos_filter(new_w.loc[i, 'similar_words'], new_w.loc[i, 'word']))

	# evasion attack 
	result = collections.defaultdict(list)
	sim = []
	path = []
	path_1 = []
	path_2 = []
	path_2_d = []
	path_3_d = []
	path_4_d = []
	path_5_d = []
	path_6_d = []
	Path = collections.defaultdict(list)
	for i in range(0, 30):
	    now = time.time()
	    sub_sample = submissions.iloc[i]['paper_text_tokens']
	    length = len(sub_sample)
	    del_list = []
	    for index, w in enumerate(sub_sample):
	        if w not in voc:
	            del_list.append(index)
	        else:
	            continue
	    sub_sample = [sub_sample[i] for i in range(0, len(sub_sample), 1) if i not in del_list]
	    '''
	    -------------------------------------Common State: 未受攻击时的正常预测---------------------------------------------
	    '''
	    n_d_sub = infer(model, sub_sample, 200)
	    d_t_sub = n_d_sub / sum(n_d_sub)
	    
	    output_new = {}
	    for index, i in enumerate(d_t_sub):
	        output_new[index] = i

	    output_new = sorted(output_new.items(),key = lambda x:x[1],reverse = True)
	    common = []
	    for j in output_new:
	        if j[1] != 0:
	            common.append(list(j))
	    target = int(random.choice(np.array(common)[:,0]))
	    while target == common[0][0]: # 排名第一的topic无需提升
	        target = int(random.choice(np.array(common)[:,0]))

	    for index, i in enumerate(output_new):
	        if i[0] == target:
	            result['Common'].append([index+1, i[0], i[1]]) # 排名，topic编号，topic分数
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_0----------------------------------------------------
	    '''
	    # Ours
	    adv_path, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_0', alpha=0.005)
	    n_d_new = infer(model, new_doc)
	    result = rank_cal(n_d_new, target, result, sim_s, 'level_0')
	    
	    # 整理攻击路径
	    Path['level_0'].append([adv_path])
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_1----------------------------------------------------
	    '''
	    adv_path_1, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_1', alpha=0.005)
	    
	    n_d_new_1 = infer(model, new_doc)
	    result = rank_cal(n_d_new_1, target, result, sim_s, 'level_1')

	    # 整理攻击路径
	    Path['level_1'].append([adv_path_1])
	    
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_2----------------------------------------------------
	    '''
	    adv_path_2, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_2', alpha=0.005)
	    
	    n_d_new_2 = infer(model, new_doc)
	    result = rank_cal(n_d_new_2, target, result, sim_s, 'level_2')

	    # 整理攻击路径
	    Path['level_2'].append([adv_path_2])

	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_2_dis----------------------------------------------------
	    '''
	    adv_path_2_d, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_2_dis', alpha=0.005)
	    
	    n_d_new_2_d = infer(model, new_doc)
	    result = rank_cal(n_d_new_2_d, target, result, sim_s, 'level_2_d')
	    # 整理攻击路径
	    Path['level_2_d'].append([adv_path_2_d])
	    
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_3----------------------------------------------------
	    '''
	    adv_path_3_d, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_3', alpha=0.005)
	    
	    n_d_new_3_d = infer(model, new_doc)
	    result = rank_cal(n_d_new_3_d, target, result, sim_s, 'level_3')
	    # 整理攻击路径
	    Path['level_3'].append([adv_path_3_d])
	    
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_4----------------------------------------------------
	    '''
	    # Ours
	    adv_path_4_d, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_4', alpha=0.005)
	    n_d_new_4_d = infer(model, new_doc)
	    result = rank_cal(n_d_new_4_d, target, result, sim_s, 'level_4')

	    # 整理攻击路径
	    Path['level_4'].append([adv_path_4_d])
	        
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_5----------------------------------------------------
	    '''
	    # Ours
	    adv_path_5_d, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_5', alpha=0.005)
	    n_d_new_5_d = infer(model, new_doc)
	    result = rank_cal(n_d_new_5_d, target, result, sim_s, 'level_5')

	    # 整理攻击路径
	    Path['level_5'].append([adv_path_5_d])
	    
	    '''
	    -------------------------------------Ours Attack: 受攻击时的预测 level_6----------------------------------------------------
	    '''
	    # Ours
	    adv_path_6_d, new_nd, new_doc, sim_s = SeakForWord(copy.deepcopy(sub_sample), target, n_d_sub, model.nzw_, length, 'Ours_level_6', alpha=0.005)
	    n_d_new_6_d = infer(model, new_doc)
	    result = rank_cal(n_d_new_6_d, target, result, sim_s, 'level_6')

	    # 整理攻击路径
	    Path['level_6'].append([adv_path_6_d])
	    
	    print('单轮时间为', time.time() - now)
	    print(result)

	print('总时间为', time.time() - now)