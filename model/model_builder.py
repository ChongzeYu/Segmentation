from model.FastSCNN import FastSCNN

def build_model(model_name, num_classes):
    if model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
