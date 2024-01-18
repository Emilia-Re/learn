def calculate_tpr_fpr(labels, scores):
    # 合并标签和分数为元组，并根据分数降序排序
    combined = sorted(zip(scores, labels), reverse=True)

    # 初始化相关指标
    tp = 0  # 真正例数
    fp = 0  # 假正例数
    fn = sum(labels)  # 假负例数，初始为所有正例数   gt T
    tn = len(labels) - fn  # 真负例数，初始为所有负例数 gt F

    # 初始化存储每个阈值的 TPR 和 FPR
    tprs = [0]
    fprs = [0]



#初始的混淆矩阵
#  gt\pred        T       F
#   T            TP     FN                   TP=0     FN=标签为真样本和
#   F            FP     TN                   FP=0     TN=标签为假样本和

#每一轮迭代更新一次混淆矩阵
# 以每个预测值为门限，左闭右开
    for score, label in combined:
        if label == 1:  # 真正例
            tp += 1
            fn -= 1
        else:  # 假正例
            fp += 1
            tn -= 1
        # 计算 TPR 和 FPR 并存储
        tprs.append(tp / (tp + fn))
        fprs.append(fp / (fp + tn))

    return tprs, fprs

def calculate_auc(tprs, fprs):
    # 使用梯形法则计算曲线下的面积
    auc = 0
    for i in range(1, len(tprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    return auc

# 测试数据
# labels = [1, 0, 1, 0, 1]
# scores = [0.8, 0.1, 0.6, 0.4, 0.9]

labels = [1, 0, 0, 0, 1, 0, 1, 0]
scores = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]

# 计算 TPR 和 FPR
tprs, fprs = calculate_tpr_fpr(labels, scores)

# 计算 AUC
auc = calculate_auc(tprs, fprs)
print(f'AUC: {auc}')
