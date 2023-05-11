import pandas as pd
from elasticsearch import Elasticsearch, helpers
import numpy as np

es = Elasticsearch(
hosts = "https://485ce3931f1e4b6393f3a256d96ba75e.eastus.azure.elastic-cloud.com:9243",
api_key = "aXhnalpZWUJuZm1CWDZDTDhBUXI6VlNEb2xkbTRRNi1IWkdBNlVXanhXdw=="
)

#### extract index from elastic
res = es.search(index="ml-model-infer", body = {
    "size":20, #max size is 10k
    'query': {
    'match_all' : {}
}
})

# print number of documents
print("documents returned:", len(res["hits"]["hits"]))
elastic_docs = res["hits"]["hits"]

fields = {}
for num, doc in enumerate(elastic_docs):
    source_data = doc["_source"]
    # iterate source data (use iteritems() for Python 2)
    for key, val in source_data.items():
        try:
            fields[key] = np.append(fields[key], val)
        except KeyError:
            fields[key] = np.array([val])

# create a Pandas DataFrame array from the fields dict
elastic_df = pd.DataFrame(fields)

print('elastic_df:', type(elastic_df), "\n")
print(elastic_df.columns) # print out the DF object's contents

lst = []
for num, doc in enumerate(elastic_docs):
    source_data = doc['_source']["categorized-ml"]['top_classes']
    lst.append(source_data)
# print(lst)

df = pd.DataFrame(lst[0])
for i in range(1, len(res["hits"]["hits"])):
    df = pd.concat([df, pd.DataFrame(lst[i])])

# df.to_csv("test.csv")
df2 = pd.DataFrame(0, index=np.arange(len(res["hits"]["hits"])), columns = ['PL_FIRE', 'PL_CRASH', 'PL_INJURY', 'PL_PROP_DAM', 'PL_FATALITY'])

c = 0
# the labels always come in different orders based on class probability
for row_val in range(0, len(df2)):
        for row_val3 in range(5):
            if c < len(df2)*5:
                if df['class_name'].iloc[c+row_val3] == "L_CRASH":
                    df2['PL_CRASH'].iloc[row_val] = df['class_probability'].iloc[c+row_val3]
                if df['class_name'].iloc[c+row_val3] == "L_FATALITY":
                    df2['PL_FATALITY'].iloc[row_val] = df['class_probability'].iloc[c+row_val3]
                if df['class_name'].iloc[c+row_val3] == "L_FIRE":
                    df2['PL_FIRE'].iloc[row_val] = df['class_probability'].iloc[c+row_val3]
                if df['class_name'].iloc[c+row_val3] == "L_INJURY":
                    df2['PL_INJURY'].iloc[row_val] = df['class_probability'].iloc[c+row_val3]
                if df['class_name'].iloc[c+row_val3] == "L_PROP_DAM":
                    df2['PL_PROP_DAM'].iloc[row_val] = df['class_probability'].iloc[c+row_val3]
        c = c + 5

#### transform it into 0,1 based on a threshold 20%
df3 = df2.copy()
df3['PL_FIRE'] = np.where((df3['PL_FIRE'] >= 0.2), 1, 0)
df3['PL_CRASH'] = np.where((df3['PL_CRASH'] >= 0.2), 1, 0)
df3['PL_INJURY'] = np.where((df3['PL_INJURY'] >= 0.2), 1, 0)
df3['PL_PROP_DAM'] = np.where((df3['PL_PROP_DAM'] >= 0.2), 1, 0)
df3['PL_FATALITY'] = np.where((df3['PL_FATALITY'] >= 0.2), 1, 0)

df4 = df3.join(elastic_df[['recordNO', 'text_field', 'L_FIRE', 'L_INJURY', 'L_CRASH', 'L_FATALITY', 'L_PROP_DAM']], how = 'outer')
print(df4)

from sklearn.metrics import f1_score
df5 = df4[~((df4['L_FIRE'] == 0) & (df4['L_CRASH'] == 0) & (df4['L_INJURY'] == 0) & (df4['L_FATALITY'] == 0) & (df4['L_PROP_DAM'] == 0))]

print('F1 score for Fire label is ', f1_score(df5['L_FIRE'],df5['PL_FIRE'], average="micro"))
print('F1 score for Crash label is ', f1_score(df5['L_CRASH'],df5['PL_CRASH'], average="micro"))
print('F1 score for Injury` label is ', f1_score(df5['L_INJURY'],df5['PL_INJURY'], average="micro"))
print('F1 score for Fatality label is ', f1_score(df5['L_FATALITY'],df5['PL_FATALITY'], average="micro"))
print('F1 score for Property Damage label is ', f1_score(df5['L_PROP_DAM'],df5['PL_PROP_DAM'], average="micro"))
# df4.to_csv("test.csv")

f1_fire = f1_score(df5['L_FIRE'],df5['PL_FIRE'], average="micro")
f1_crash = f1_score(df5['L_CRASH'],df5['PL_CRASH'], average="micro")
f1_injury = f1_score(df5['L_INJURY'],df5['PL_INJURY'], average="micro")
f1_fatality = f1_score(df5['L_FATALITY'],df5['PL_FATALITY'], average="micro")
f1_prop_dam =  f1_score(df5['L_PROP_DAM'],df5['PL_PROP_DAM'], average="micro")

from azure.storage.blob import BlobServiceClient
LATEST_DATA_CONTAINER_NAME = "testcontainer01"
EMAIL_CONTAINER_NAME = "emails-container"

blobService = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=mlpipelinetest01;AccountKey=0WX38QP+B4uL0RC6B8VGwB3qEN9Z4HTuQCBDYz/SSndLBoz2RlS8CqhC6uouR4eqS9Goo2P29GTo+AStHa+Qgw==;EndpointSuffix=core.windows.net")

print("\nList blobs in the container")
LatestDataContainer = blobService.get_container_client(LATEST_DATA_CONTAINER_NAME)
print(LatestDataContainer.list_blobs())

blob_client = LatestDataContainer.get_blob_client(blob='train_data.csv')


EmailsContainer = blobService.get_container_client(EMAIL_CONTAINER_NAME)
print(EmailsContainer.list_blobs())

df6 = {}
from datetime import date
if (f1_fire < 0.9) | (f1_crash < 0.9) | (f1_injury < 0.9) | (f1_fatality < 0.9) | (f1_prop_dam < 0.9):

    # check which scores are below 90% from the labels
    if f1_fire < 0.9 & len(df5[df5['L_FIRE'] > 0]) > 100:
        # add text field, and add actual labels in one file
        df6 = df5[['text_field', 'L_FIRE', 'L_CRASH', 'L_INJURY', 'L_PROP_DAM', 'L_FATALITY']][df['L_FIRE'] > 0]

    if f1_crash < 0.9 & len(df5[df5['L_CRASH'] > 0]) > 100:
        # add text field, and add actual labels in one file
        df6 = pd.concat([df6, df5[['text_field', 'L_FIRE', 'L_CRASH', 'L_INJURY', 'L_PROP_DAM', 'L_FATALITY']][df['L_CRASH'] > 0]])

    if f1_injury < 0.9 & len(df5[df5['L_INJURY'] > 0]) > 100:
        # add text field, and add actual labels in one file
        df6 = pd.concat([df6, df5[['text_field', 'L_FIRE', 'L_CRASH', 'L_INJURY', 'L_PROP_DAM', 'L_FATALITY']][df['L_CRASH'] > 0]])

    if f1_fatality < 0.9 & len(df5[df5['L_INJURY'] > 0]) > 100:
        # add text field, and add actual labels in one file
        df6 = pd.concat([df6, df5[['text_field', 'L_FIRE', 'L_CRASH', 'L_INJURY', 'L_PROP_DAM', 'L_FATALITY']][df['L_CRASH'] > 0]])

    blob_client.upload_blob(df6.to_csv())
    today = date.today()
    # print("Today's date:", today)
    blob_client02 = EmailsContainer.get_blob_client(blob=f'email_{today}.txt')
    # send file to azure blob
    email_content = f'The previous accuracy of the model is 90% and the current accuracy of the model for \n ' \
                    f'for fire label {f1_fire} \n' \
                    f'for crash label {f1_crash} \n' \
                    f'for injury label {f1_injury} \n' \
                    f'for property damage {f1_prop_dam} \n' \
                    f'for fatality {f1_fatality} \n'

    blob_client02.upload_blob(email_content)
else:
    # Upload the created file
    today = date.today()
    #print("Today's date:", today)
    blob_client02 = EmailsContainer.get_blob_client(blob=f'email_{today}.txt')
    #send file to azure blob
    email_content = f'The previous accuracy of the model is 90% and the current accuracy of the model for \n ' \
                    f'for fire label {f1_fire} \n' \
                    f'for crash label {f1_crash} \n' \
                    f'for injury label {f1_injury} \n' \
                    f'for property damage {f1_prop_dam} \n' \
                    f'for fatality {f1_fatality} \n' \
                    f'The model does not need to be improved.'

    blob_client02.upload_blob(email_content)
