import sklearn.datasets as dts
import sklearn.tree as tr
import graphviz

data = dts.load_iris()
x_data, y_data = dts.load_iris(return_X_y=True)
# 对鸢尾花数据进行分类，利用决策树
dec = tr.DecisionTreeClassifier(max_depth=5, random_state=42)
dec.fit(x_data, y_data)
data_tree = tr.export_graphviz(
    dec,
    out_file=None,
    feature_names=data.feature_names,
    class_names=['0', '1', '2'],
    rounded=True)
grap = graphviz.Source(data_tree)
grap.render('load_iris.pdf')
# 显示绘图
grap.view()
