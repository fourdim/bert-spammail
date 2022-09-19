import sys
sys.path.append(".") 

from src.detector import Detector

d = Detector()
d.load_model('./model/weights')
result = d.predict('[GitHub] A personal access token has been added to your account', 'A personal access token (gh cli) with admin:org and repo scopes was recently added to your account. Visit https://github.com/settings/tokens for more information.')
print(result.with_threshold(0).is_spam())
