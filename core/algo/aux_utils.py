import tensorflow as tf


def quantile_loss(q, name=None):  # 分位数损失函数
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), name=name)
    # 为损失函数设置唯一名称
    if name:
        loss.__name__ = name
    return loss


def get_quantile_metrics(q=0.1):  # 获取分位数指标
    if q < 0 or q > 1:
        raise ValueError("Quantile should be between 0 and 1")
    if q < 1:
        first_name = "q{}_loss".format(int(q*100))
        second_name = "q{}_loss".format(int((1-q)*100))
        return {
            "mae": "mean_absolute_error",
            first_name: quantile_loss(q, name=first_name),
            second_name: quantile_loss(1 - q, name=second_name)}

    else:
        first_name = "q{}_loss".format(int(q*100))
        return {
            "mae": "mean_absolute_error",
            first_name: quantile_loss(q, name="q10_loss")}


def get_quantile_loss(q=0.1):  # 获取分位数损失函数
    if q < 0 or q > 1:
        raise ValueError("Quantile should be between 0 and 1")
    if q < 1:
        first_name = "q{}_loss".format(int(q*100))
        second_name = "q{}_loss".format(int((1-q)*100))
        return {
            "quantile_loss": quantile_loss,
            first_name: quantile_loss(q, name=first_name),
            second_name: quantile_loss(1 - q, name=second_name)}

    else:
        first_name = "q{}_loss".format(int(q*100))
        return {
            "quantile_loss": quantile_loss,
            first_name: quantile_loss(q, name="q10_loss")}


if __name__ == '__main__':
    metric = get_quantile_metrics(0.1)
    print(metric)
    print(list(metric.values()))