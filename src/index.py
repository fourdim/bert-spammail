class DetectorIndex:
    def __init__(self, index):
        self.threshold = 0
        self.index = index

    def with_threshold(self, threshold):
        self.threshold = threshold
        return self

    def is_spam(self):
        if self.index < self.threshold:
            return False
        else:
            return True
