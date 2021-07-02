import shap
shap.initjs()

def treeMLAlgosExplainer(model, trainData, testData, features, valueAt):
    explainer = shap.TreeExplainer(model)
    shap_values_train = explainer.shap_values(trainData)
    shap_values_test = explainer.shap_values(testData)
    
    shap.summary_plot(shap_values_train, trainData)
    shap.force_plot(explainer.expected_value, shap_values_test[valueAt], testData[valueAt])

def deepMLAlgosExplainer(model, trainData, testData, features, mlAlgo):
    pass

def otherMLAlgosExplainer(model, trainData, testData, features, valueAt):
    explainer = shap.KernelExplainer(model.predict, testData)
    shap_values = explainer.shap_values(testData)
    
    shap.summary_plot(shap_values, testData)