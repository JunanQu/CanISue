"""
***Follow these instructions to set env variables. Only run once***

Sign up here: https://case.law/user/register/

Click Account, get your API key.

Open terminal

vim ~/.bash_profile

press i to enter insert mode

go to the bottom of the file and add these three lines:

export CASELAW_APIKEY = 'your_api_key'

press escape and the type ':wq!' to save and exit

run source ~/.bash_profile to update your environment

You will need to shut down the jupyter notebook/restart your terminal to see the effects.
"""

import requests
import os
def get_request_caselaw(url):
    """
    url: string
    Requires user to have followed instructions setting bash profile above
    Returns: request response object
    """
    return requests.get(
    url,
    headers={'Authorization': 'Token ' + os.environ['CASELAW_APIKEY']}
    )