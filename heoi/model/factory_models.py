
class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):

        model = None
        # print(model_name)
        if model_name == 'Dense_V1':
            from model.model_DenseInter_v1_one import CustomModel
            model = CustomModel(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model
