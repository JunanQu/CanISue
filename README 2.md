# CanISue


To run the app locally, run the following command
------
```
export APP_SETTINGS=config.DevelopmentConfig; 
export DATABASE_URL=postgresql://localhost/my_app_db;
export CASELAW_APIKEY=9c9cb289a556cf122fa89f0979c174ede5705ab8;
python3 app.py

```

To deploy on AWS

```
ansible -m ping webservers --private-key=a4keypair.pem;
ansible-playbook -v site.yml
```