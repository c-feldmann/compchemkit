from sklearn.model_selection import GridSearchCV


class ModelGenerator:
    def __init__(self, classifier, parameter_dict, name=None, sampler=None, score="balanced_accuracy",
                 n_jobs=-1):
        self.classifier = classifier
        self.parameter_dict = parameter_dict
        self.name = name
        if sampler is None:
            self.sampler = "cv"
        else:
            self.sampler = sampler
        self.score = score
        self.n_jobs = n_jobs

    def build_model(self):
        model = GridSearchCV(self.classifier,
                             param_grid=self.parameter_dict,
                             cv=self.sampler,
                             scoring=[self.score],
                             refit=self.score,
                             n_jobs=self.n_jobs)
        return model
