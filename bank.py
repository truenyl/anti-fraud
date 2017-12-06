# -*- coding: utf-8 -*-

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.sql.types import StructField, StructType, FloatType, StringType, DoubleType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import ChiSqSelector
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.functions import rand
from pyspark.mllib.classification import SVMWithSGD
from pyspark.sql import SparkSession, SQLContext #2.1
import math
import numpy as np

#if __name__ == "__main__":
spark = SparkSession.builder.appName("modeling").getOrCreate()

# 转LabeledPoint格式
def ToFL(lines):
    values = [float(x) for x in lines]
    return LabeledPoint(values[0:-1],values[-1])

# 数据类型转float
def ToF(lines):
    values = [float(x) for x in lines]
    return values

# fill null
def isnull(x):
    if (x == 'null'):
        return '0'
    else:
        return x

def fillnanull(x):
    x = map(isnull,x)
    return x

# 离散化
def ToDiscrete(lines):
    values = [math.floor(float(x)) for x in lines]
    return values

sc = spark.sparkContext
# 读csv为RDD
data = sc.textFile(r"/home/truenyl/aws/data/creditcard.csv").map(lambda x:x.split(',')).map(fillnanull)  
head = data.take(1)[0] 
# 去掉列名行
data0 = data.filter(lambda lines: lines[0] != 'Time')


'''
#databricks包读csv为DataFrame
data = spark.load(source = 'com.databricks.spark.csv',header = 'true',path = "/home/truenyl/aws/data/creditcard.csv")
data = data.drop('Time')
head = data.columns
data = data.map(fillnanull).map(ToF)
'''
# RDD转为DataFrame
field = [StructField(field_name,DoubleType(),True) for field_name in head]
schema = StructType(field)
sqlContext = SQLContext(sc)
data_new = sqlContext.createDataFrame(data.map(ToF),schema)
training,test = data_new.randomSplit([0.6,0.4],seed = 24)

# 卡方特征选择，前30变量
# r = ChiSqSelector(30).fit(data_new.map(ToDiscrete)).transform(data_new.map(ToDiscrete).map(lambda x: x.features))

# 自动选取最优参数类
class modeling:
    # dataset - 训练集， estimator - 模型， estimatorParamMaps - 模型所有参数组合， samplingrates - 负样本抽样率list， numfolds - 交叉验证折数
    def _fit(self, dataset, estimator, estimatorParamMaps, samplingrates, numfolds = 5):
        all_list = dataset.columns
        all_list.remove('Class') #所有特征列名
        assembler = VectorAssembler().setInputCols(all_list).setOutputCol("features_vector") #特征列转换为一列向量
        labelIndexer = StringIndexer(inputCol="Class", outputCol="label") #统一标签列名称为label
        featureIndexer = VectorIndexer(inputCol="features_vector", outputCol="features", maxCategories=10) #统一特征向量列名称，不同值数量小于10视作离散变量编号
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, estimator]) #机器学习流建模，三部分整合
        dataset = assembler.transform(dataset) #训练集生成特征向量列
        best_epm, best_sampling, metricsX = self.Cross_Validation(dataset, estimator, estimatorParamMaps, samplingrates, numfolds) #交叉验证
        bestModel = pipeline.fit(dataset.sampleBy("Class", fractions = {1.0: 1.0, 0.0: best_sampling}),best_epm) # fit最优模型并输出
        return bestModel, best_epm, best_sampling
    def Cross_Validation(self, dataset, estimator, estimatorParamMaps, samplingrates, numFolds):
        est = estimator
        epm = estimatorParamMaps
        numModels = len(epm) #参数组合长度
        sam = samplingrates
        nFolds = numFolds
        h = 1.0 / nFolds
        metrics = np.zeros(numModels * len(sam))
        for k in range(len(sam)):
            training = dataset.sampleBy("Class", fractions = {1.0: 1.0, 0.0: sam[k]}, seed = 0).repartition(1)
            df = dataset.select('*', rand(0).alias('_rand')) #加入一列随机数
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df['_rand'] >= validateLB) & (df['_rand'] < validateUB) #按随机数分折
                validation = df.filter(condition)
                train = df.filter(~condition)
                for j in range(numModels):
                    model = est.fit(train, epm[j])
                    predictions = model.transform(validation, epm[j])
                    tp = predictions.map(lambda x: x.prediction == 1 and x.label == 1).filter(lambda f: f == True).count() #true positive
                    fp = predictions.map(lambda x: x.prediction == 1 and x.label == 0).filter(lambda f: f == True).count() #false positive
                    fn = predictions.map(lambda x: x.prediction == 0 and x.label == 1).filter(lambda f: f == True).count() #false negetive
                    if (tp + fp == 0 or tp == 0):
                        metrics = 0
                    else:
                        p = float(tp) / float(tp + fp)
                        precision = p / (p + (1 - p) * 5.0)       #此处5.0为将抽样率1:4下的数据集精准率转换为1:20数据集上的情况，即（1/4）/（1/20），根据训练集和实际测试情况调整
                        recall = float(tp) / float(tp + fn)       #召回
                        metrics = 2 / ((1 / recall) + (1 / precision))    #F1-score
                    metricsX[k * numModels + j] += metrics
        bestIndex = np.argmax(metrics)
        return epm[bestIndex], best_sampling, metricsX

# 设置默认分类器
rf = RandomForestClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction", cacheNodeIds=True, featureSubsetStrategy="all")
gbt = GBTClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", cacheNodeIds=True)

# 构造参数网格
paramGrid_rf = ParamGridBuilder().addGrid(rf.maxDepth,[8,12,15]).addGrid(rf.minInfoGain,[0.0,0.001]).addGrid(rf.minInstancesPerNode,[1,2,3]).addGrid(rf.numTrees, [20,50,100,120]).build()
paramGrid_gbt = ParamGridBuilder().addGrid(gbt.maxDepth,[5,8,10]).addGrid(gbt.minInfoGain,[0.0,0.001]).addGrid(gbt.minInstancesPerNode,[1,2,3]).addGrid(gbt.maxIter, [100,150,200]).addGrid(gbt.stepSize, [0.01,0.1]).build()

# modeling类调用方式
bestModel_rf, best_epm_rf, best_sampling_rf = modeling()._fit(training, rf, paramGrid_rf, [0.2,0.5,0.8], 5)
bestModel_gbt, best_epm_gbt, best_sampling_gbt = modeling()._fit(training, gbt, paramGrid_gbt, [0.2,0.5,0.8], 5)

# 预测
predictions_rf = bestModel_rf.transform(test)
predictions_gbt = bestModel_gbt.transform(test)

# 混淆矩阵
predictions_rf.groupBy('label','predictions').count().show()
predictions_gbt.groupBy('label','predictions').count().show()

# SVM
training_svc = training.map(lambda x:LabeledPoint(x[0],x[1:]))
sv = SVMWithSGD.train(training_svc,iterations = 100,step = 0.1,regParam = 0.01)
test_svc = training.map(lambda x:LabeledPoint(x[0],x[1:]))
predictions = test_svc.map(lambda x:(x.TARGET,float(sv.predict(x.features))))