# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ibm_db_dbi
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import jaydebeapi
from azure.storage.blob import BlobServiceClient
import ibm_db
from wordcloud import WordCloud, STOPWORDS
import argparse
import json
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser("fetching")
parser.add_argument("--real_data", type=str, help="Path to fetched data")
args = parser.parse_args()

CATEGORIES = ["L_CRASH", "L_FIRE", "L_PROP_DAM", "L_INJURY", "L_FATALITY"]

COMMON_WORDS = ['vehicle',
'phone number',
'case',
'customer',
'email address',
'email sent',
'reason code',
'advised customer',
'unknown',
'customers',
'owner',
'advised owner',
'phone',
'default',
'dealer',
'date',
'contact']

SAMPLE_SIZE = 1000
CAIR_SQL_PREFIX = """select cat.I_REQ_NO,
       cat.I_KEY, 
       replace(replace(cat.L_CRASH, 'Y','1'),'N','0') L_CRASH,
       replace(replace(cat.L_FIRE, 'Y','1'),'N','0') L_FIRE, 
       replace(replace(cat.L_PROP_DAM, 'Y','1'),'N','0') L_PROP_DAM, 
       case when cat.I_INJURY <= 0 then '0' else '1' end l_injury,
       case when cat.I_FATALITY <= 0 then '0' else '1' end l_fatality
  from db2dap.edapcat cat 
  where cat.C_SOURCE = 'CAIR'
    and cat.{%CRASH_COND%} """
CAIR_SQL = CAIR_SQL_PREFIX + \
           "and cat.{%FIRE_COND%} " + \
           "and cat.{%PROP_DAM_COND%} " + \
           "and cat.{%FATALITY_COND%} " + \
           "and cat.{%INJURY_COND%} " + \
           "fetch first " + str(SAMPLE_SIZE) + " rows only with ur"

VOQ_SQL_PREFIX = """select DISTINCT
       cat.I_KEY, 
       replace(replace(cat.L_CRASH, 'Y','1'),'N','0') L_CRASH,
       replace(replace(cat.L_FIRE, 'Y','1'),'N','0') L_FIRE, 
       replace(replace(cat.L_PROP_DAM, 'Y','1'),'N','0') L_PROP_DAM, 
       case when cat.I_INJURY <= 0 then '0' else '1' end l_injury,
       case when cat.I_FATALITY <= 0 then '0' else '1' end l_fatality
  from db2dap.edapcat cat 
  where cat.C_SOURCE = 'VOQ'
    and cat.{%CRASH_COND%} """
VOQ_SQL = VOQ_SQL_PREFIX + \
           "and cat.{%FIRE_COND%} " + \
           "and cat.{%PROP_DAM_COND%} " + \
           "and cat.{%FATALITY_COND%} " + \
           "and cat.{%INJURY_COND%} " + \
           "fetch first " + str(SAMPLE_SIZE) + " rows only with ur"

SAMPLING_LIST = [[1, 0, 0, 0, 0],
[1, 1, 0, 0, 0],
[1, 1, 1, 0, 0],
[1, 1, 1, 1, 0],
[1, 1, 1, 1, 1],
[1, 0, 1, 0, 0],
[1, 0, 1, 1, 0],
[1, 0, 1, 1, 1],
[1, 0, 0, 1, 0],
[1, 0, 0, 1, 1],
[0, 1, 0, 0, 0],
[0, 1, 1, 0, 0],
[0, 1, 1, 1, 0],
[0, 1, 1, 1, 1],
[0, 1, 0, 1, 0],
[0, 1, 0, 1, 1],
[0, 0, 1, 0, 0],
[0, 0, 1, 1, 0],
[0, 0, 1, 1, 1],
[0, 0, 1, 0, 1],
[0, 0, 0, 1, 0],
[0, 0, 0, 1, 1],
[0, 0, 0, 0, 1]]

SAMPLING_LIST_SMALL = [[1, 0, 0, 0, 0],
[1, 1, 0, 0, 0]]

BASE_DIR = "C:/Elastic/requests_data/CAIRs_Categorized/all data"
CONCATENATED_TEXT_SQL  = """SELECT int(main.casenumber) CASE_NUMBER,
       concat(concat(concat(concat(concat(concat(concat(concat(concat(concat(nvl(main.subject, ' '), ' '), 
              concat(COALESCE(nvl(main.cc_customeranomalydescription__c,' '), ''),' ')),
                concat(COALESCE(nvl(main.cc_anomalydeclaredbycustomerreason1__c,' '), '' ), ' ')),
                  concat(COALESCE(nvl(main.cc_anomalydeclaredbycustomerreason2__c,' '), '' ), ' ')),
                    concat(COALESCE(nvl(main.cc_anomalydeclaredbycustomerreason3__c, ' '), '' ),' ')),
                      concat(nvl(main.description,' '), ' ')),
                        concat(nvl(narrative, ' '), ' ')),
                          concat(nvl(email_subject, ' '), ' ')),
                            concat(nvl(email_body, ' '), ' ')),
                              concat(nvl(chatbody, ' ' ), ' ')) CONCATENATED_TEXT

   FROM   GCCRA.CASE main

        LEFT OUTER JOIN GCCRA.ASSET vhcl ON main.assetid = vhcl.id  

        LEFT OUTER JOIN GCCRA.ACCOUNT cust ON main.accountid= cust.id  

        LEFT OUTER JOIN TABLE (

        SELECT main.casenumber case_number, min(email.createddate) createddate, max(email.lastmodifieddate) lastmodifieddate,

               xmlserialize( xmlagg(xmlconcat(xmltext(REGEXP_REPLACE(email.textbody,'[^a-zA-Z0-9,\.;?!\s]','')), xmltext('')) ) as clob) as email_body,

               xmlserialize( xmlagg(xmlconcat(xmltext(REGEXP_REPLACE(email.subject,'[^a-zA-Z0-9,\.;?!\s]','')), xmltext('')) ) as clob) as email_subject

            FROM GCCRA.CASE main INNER JOIN GCCRA.EMAIL_MESSAGE email ON main.id = email.parentid 

        GROUP BY main.casenumber) email

            ON main.casenumber = email.case_number



        LEFT OUTER JOIN TABLE (

       SELECT main.casenumber case_number, min(cmnt.createddate) createddate, max(cmnt.lastmodifieddate) lastmodifieddate,

               --xmlserialize( xmlagg(xmlconcat(xmltext(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(cmnt.COMMENTBODY,chr(26),''),chr(5),''),chr(3),'$#$'),chr(6),''),chr(4),''),chr(18),''),chr(20),''),chr(146),''),chr(194),''),chr(168),''),chr(147),''),chr(148),''),chr(133),''),chr(149),''),chr(150),''),chr(160),''),chr(151),''),chr(174),''),chr(153),''),chr(188),''),chr(195),''),chr(157),''),chr(169),''),chr(179),''),chr(161),''),chr(186),''),chr(177),''),chr(176),''),chr(173),''),chr(173),'')), xmltext('')) ) as clob) as narrative
               xmlserialize( xmlagg(xmlconcat(xmltext(REGEXP_REPLACE(cmnt.COMMENTBODY,'[^a-zA-Z0-9,\.;?!\s]','')), xmltext('')) ) as clob) as narrative

            FROM GCCRA.CASE main INNER JOIN GCCRA.CASE_COMMENT cmnt ON main.id = cmnt.parentid

        GROUP BY main.casenumber) nrtv

            ON main.casenumber = nrtv.case_number

        LEFT OUTER JOIN TABLE (

        SELECT main.casenumber case_number,  min(chat.createddate) createddate, max(chat.lastmodifieddate) lastmodifieddate,

               xmlserialize( xmlagg(xmlconcat(xmltext(REGEXP_REPLACE(chat.body,'[^a-zA-Z0-9,\.;?!\s]','')), xmltext('')) ) as clob) as chatbody

            FROM GCCRA.CASE main INNER JOIN GCCRA.LIVE_CHAT_TRANSCRIPT chat ON main.id = chat.caseid

        GROUP BY main.casenumber) chat

            ON main.casenumber = chat.case_number

   WHERE main.cc_lob__c = 'CAC'
   and main.CASENUMBER in ({IDS}) with ur"""

VOQ_TEXT_SQL  = """select e.C_NVQ_ODINO ODI_NO,
                         case when e.C_NVQ_CRASH in ('Y','1') then 1 else 0 end X_CRASH,
                         case when e.C_NVQ_FIRE in ('Y','1') then 1 else 0 end X_FIRE,         
                         case when e.I_NVQ_DEATHS > 0 then 1 else 0 end X_FATALITY,
                         case when e.I_NVQ_INJURED > 0 then 1 else 0  end X_INJURY,
                         e.X_NVQ_CSUMARY VOQ_SUMMARY, 
                         e.C_NVQ_COMPDESC VOQ_COMP                 
                    from DB2DAP.EDAPNVQ e where e.C_NVQ_ODINO in ({IDS}) with ur"""

PROD_DB = f"DATABASE=DAPIS;HOSTNAME=srvr2274.dbms.chrysler.com;PORT=22740;PROTOCOL=TCPIP;UID=T0075C8;PWD=e24aDGLU@123;"
DEV_DB = f"DATABASE=DAPIS;HOSTNAME=srvr2197.dbms.chrysler.com;PORT=21970;PROTOCOL=TCPIP;UID=db2dap;PWD=dcx1971;"
SALESFORCE_DB = f"DATABASE=SFDCODPA;HOSTNAME=srvr3059.dbms.chrysler.com;PORT=30590;PROTOCOL=TCPIP;UID=db2dap;PWD=WULxA7QY7W5WY1#;"

def get_text_from_file(file_name):
    if file_name != "(null)":
        start = "CAIR_reports\\"
        end = "||TextData"
        parent_path = "C:/Elastic/requests_data/CAIRs_Categorized/all data/CAIR_reports"
        file_name = (file_name.split(start))[1].split(end)[0]

        full_file_path = os.path.join(parent_path, file_name)
        with open(full_file_path,'r') as f:
            contents = f.readlines()
        return ''.join(contents)
    else:
        return ''

def plot_dist(series_to_plot, title, x_label, y_label):
    figsize(7, 5)
    plt.hist(series_to_plot, color='blue', edgecolor='black', bins=int(45 / 1))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("C:/Elastic/requests_data/CAIRs_Categorized/all data/test.jpg")

def prepare_cair_data(sql):
    connection = ibm_db.connect(
        PROD_DB,
        "",
        ""
    )
    print("Opened DB Connection")
    index = ['I_REQ_NO','CASE_NUMBER', 'L_CRASH', 'L_FIRE', 'L_PROP_DAM', 'L_INJURY', 'L_FATALITY' ]
    stmt = ibm_db.prepare(connection, sql)
    ibm_db.execute(stmt)
    data = []
    row = ibm_db.fetch_tuple(stmt)
    while (row):
        data.append(row)
        row = ibm_db.fetch_tuple(stmt)

    df = pd.DataFrame(data, columns=index)

    ibm_db.close(connection)
    print("Closed DB Connection")
    return df

def prepare_voq_data(sql):
    connection = ibm_db.connect(
        PROD_DB,
        "",
        ""
    )
    print("Opened DB Connection")
    index = ['ODI_NO', 'L_CRASH', 'L_FIRE', 'L_PROP_DAM', 'L_INJURY', 'L_FATALITY' ]
    stmt = ibm_db.prepare(connection, sql)
    ibm_db.execute(stmt)
    data = []
    row = ibm_db.fetch_tuple(stmt)
    while (row):
        data.append(row)
        row = ibm_db.fetch_tuple(stmt)

    df = pd.DataFrame(data, columns=index)

    ibm_db.close(connection)
    print("Closed DB Connection")
    return df

def prepare_text_df():
    df = pd.read_csv('C:/Elastic/requests_data\CAIRs_Categorized/all data/CAIR_reports_list.csv')
    df['DESCRIPTION_TEXT'] = df['DESCRIPTION'].apply(get_text_from_file)
    df['NARRATIVE_TEXT'] = df['NARRATIVE'].apply(get_text_from_file)
    df.fillna(" ", inplace=True)

    df['CONCATENATED'] = df['SUBJECT'] \
                         + " " + df['ANOMALY_DESC'] \
                         + " " + df['REASON1'] \
                         + " " + df['REASON2'] \
                         + " " + df['REASON3'] \
                         + " " + df['DESCRIPTION_TEXT'] \
                         + " " + df['NARRATIVE_TEXT'] \
                         + " " + df['EMAIL_BODY'] \
                         + " " + df['CHATBODY']
    df = df[['CONCATENATED', "L_CRASH", "L_FIRE", "L_PROP_DAM", "L_INJURY", "L_FATALITY"]]
    plot_dist(df['L_PROP_DAM'], "Property Damage Dist.", "Property Damage", "Values Cnt.")
    df.to_csv("C:/Elastic/requests_data\CAIRs_Categorized/all data/CAIR_labels_ready.csv")

    print(df.iloc[0].CONCATENATED)

def generate_cair_list_df():
    # df_crash = get_cair_data([1, 0, 0, 0, 0])
    # final_df = df_crash
    # print("inside generate cair list df: ")
    # print(final_df.head(5))
    final_df = pd.DataFrame()
    for sample_list in SAMPLING_LIST_SMALL:
        sample_df = get_cair_data(sample_list)
        print("inside generate cair list df: ")
        print(sample_df.head(5))
        final_df = pd.concat([final_df, sample_df], ignore_index=True)
    return final_df

def generate_voq_list_df():
    final_df = pd.DataFrame()
    for sample_list in SAMPLING_LIST:
        sample_df = get_voq_data(sample_list)
        print("inside generate cair list df: ")
        print(sample_df.head(5))
        final_df = pd.concat([final_df, sample_df], ignore_index=True)
    return final_df


def get_cair_data(logic_list):
    final_sql = CAIR_SQL
    default_condition_dict = {"0": ("{%CRASH_COND%}","L_CRASH in ('N','0')"),
                      "1": ("{%FIRE_COND%}","L_FIRE in ('N','0')"),
                      "2": ("{%PROP_DAM_COND%}", "L_PROP_DAM in ('N','0')"),
                      "3": ("{%FATALITY_COND%}", "I_FATALITY <= 0"),
                      "4": ("{%INJURY_COND%}", "I_INJURY <= 0")
                      }
    set_condition_dict = {"0": ("{%CRASH_COND%}","L_CRASH in ('Y','1')"),
                      "1": ("{%FIRE_COND%}","L_FIRE in ('Y','1')"),
                      "2": ("{%PROP_DAM_COND%}", "L_PROP_DAM in ('Y','1')"),
                      "3": ("{%FATALITY_COND%}", "I_FATALITY > 0"),
                      "4": ("{%INJURY_COND%}", "I_INJURY > 0")
                      }
    for i in range(0,5):
        if logic_list[i]:
            final_sql = final_sql.replace(set_condition_dict[str(i)][0], set_condition_dict[str(i)][1])
        else:
            final_sql = final_sql.replace(default_condition_dict[str(i)][0],default_condition_dict[str(i)][1])

    print(" final sql: " + final_sql)
    df = prepare_cair_data(final_sql)
    df['CASE_NUMBER'] = "'" + df['CASE_NUMBER'].map(str) + "'"
    return df

def get_voq_data(logic_list):
    final_sql = VOQ_SQL
    default_condition_dict = {"0": ("{%CRASH_COND%}","L_CRASH in ('N','0')"),
                      "1": ("{%FIRE_COND%}","L_FIRE in ('N','0')"),
                      "2": ("{%PROP_DAM_COND%}", "L_PROP_DAM in ('N','0')"),
                      "3": ("{%FATALITY_COND%}", "I_FATALITY <= 0"),
                      "4": ("{%INJURY_COND%}", "I_INJURY <= 0")
                      }
    set_condition_dict = {"0": ("{%CRASH_COND%}","L_CRASH in ('Y','1')"),
                      "1": ("{%FIRE_COND%}","L_FIRE in ('Y','1')"),
                      "2": ("{%PROP_DAM_COND%}", "L_PROP_DAM in ('Y','1')"),
                      "3": ("{%FATALITY_COND%}", "I_FATALITY > 0"),
                      "4": ("{%INJURY_COND%}", "I_INJURY > 0")
                      }
    for i in range(0,5):
        if logic_list[i]:
            final_sql = final_sql.replace(set_condition_dict[str(i)][0], set_condition_dict[str(i)][1])
        else:
            final_sql = final_sql.replace(default_condition_dict[str(i)][0],default_condition_dict[str(i)][1])

    print(" final sql: " + final_sql)
    df = prepare_voq_data(final_sql)
    return df

def prepare_cair_text(sql):
    connection = ibm_db.connect(
        SALESFORCE_DB,
        "",
        ""
    )
    print("Opened DB Connection")
    index = ['CASE_NUMBER', 'CONCATENATED_TEXT']
    stmt = ibm_db.prepare(connection, sql)
    ibm_db.execute(stmt)
    data = []
    row = ibm_db.fetch_tuple(stmt)
    while (row):
        data.append(row)
        row = ibm_db.fetch_tuple(stmt)

    df = pd.DataFrame(data, columns=index)
    # print(df.head(3))

    ibm_db.close(connection)
    print("Closed DB Connection")
    return df

def prepare_voq_text(sql):
    connection = ibm_db.connect(
        PROD_DB,
        "",
        ""
    )
    print("Opened DB Connection")
    index = ['ODI_NO', 'X_CRASH','X_FIRE','X_FATALITY','X_INJURY','VOQ_SUMMARY', 'VOQ_COMP']
    stmt = ibm_db.prepare(connection, sql)
    ibm_db.execute(stmt)
    data = []
    row = ibm_db.fetch_tuple(stmt)
    while (row):
        data.append(row)
        row = ibm_db.fetch_tuple(stmt)

    df = pd.DataFrame(data, columns=index)
    # print(df.head(3))

    ibm_db.close(connection)
    print("Closed DB Connection")
    return df

def generate_word_cloud(df):
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in df.CONCATENATED_TEXT:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

        for word in COMMON_WORDS:
            if word in comment_words:
                comment_words = comment_words.replace(word, "")

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig("C:/Elastic/requests_data/CAIRs_Categorized/all data/word_cloud.jpg")
def generate_documents_list(df):

    documents_list = []
    doc_dict_classes = []
    for index, row in df.iterrows():
        doc_dict = {}
        doc_dict['location'] = row['CASE_NUMBER'] + '.txt'
        main_path = 'C:\Elastic/requests_data/CAIRs_Categorized/all data/cair_reports_for_model'
        file_name = row['CASE_NUMBER'] + '.txt'
        full_path = os.path.join(main_path,file_name)
        with open(full_path, 'w',encoding="utf-8") as f:
            f.write(row['CONCATENATED_TEXT'])
        doc_dict['language'] = 'en-us'
        cat_list = []
        for category in CATEGORIES:
            if row[category] == 1:
                cat_dict = {}
                cat_dict['Category'] = category
                cat_list.append(cat_dict)
        if len(cat_list) == 0:
            cat_dict = {}
            cat_dict['Category'] = "NONE"
            cat_list.append(cat_dict)
        doc_dict['classes'] = cat_list
        documents_list.append(doc_dict)


    return documents_list


def generate_json_file(df):
    classes = ["L_CRASH", "L_FIRE", "L_PROP_DAM", "L_INJURY", "L_FATALITY", "NONE"]
    categories_dict = {
        "projectFileVersion": "2022-05-01",
        "stringIndexType": "Utf16CodeUnit",
        "metadata":
            {
                "projectKind": "CustomMultiLabelClassification",
                "storageInputContainerName": "cair-data",
                "projectName": "cair_classification",
                "multilingual": False,
                "description": "multi label cair text classification",
                "language": "en",
                "settings": {}
            },
    }
    classes_dict_list = []
    for category in classes:
        cat_dict = {}
        cat_dict['category'] = category
        classes_dict_list.append(cat_dict)

    assets_dict = {
        "projectKind": "customMultiLabelClassification",
        "classes": classes_dict_list
    }
    doc_list = generate_documents_list(df)
    assets_dict['documents'] = doc_list
    categories_dict['assets'] = assets_dict
    json_object = json.dumps(categories_dict)
    # Writing to sample.json
    with open("C:/Elastic/requests_data/CAIRs_Categorized/all data/cair_sample.json", "w") as outfile:
        outfile.write(json_object)
def get_current_date_components():
    todays_date = date.today()
    prefix = str(todays_date.year) + "_" + str(todays_date.month) + "_" + str(todays_date.day)
    return prefix


def generate_CAIR_data(todays_components):
    cair_list_file_name = "cair_list_from_python_" + todays_components + ".csv"
    cair_list_full_file_path = os.path.join(BASE_DIR, cair_list_file_name)
    text_file_name = "cair_text_" + todays_components + ".csv"
    text_full_file_path = os.path.join(BASE_DIR, text_file_name)
    df = generate_cair_list_df()
    df.to_csv(cair_list_full_file_path)
    df = pd.read_csv(cair_list_full_file_path)
    case_number_list = df['CASE_NUMBER']
    case_numbers = ",".join(df['CASE_NUMBER'])
    sql = CONCATENATED_TEXT_SQL.replace("{IDS}", case_numbers)
    df = prepare_cair_text(sql)
    df['CASE_NUMBER'] = "'" + df['CASE_NUMBER'].map(str) + "'"
    df.to_csv(text_full_file_path)
    df_cair_list = pd.read_csv(cair_list_full_file_path)
    df_cair_text = pd.read_csv(text_full_file_path)
    df_full_cair_list = pd.merge(df_cair_list, df_cair_text, how='inner', left_on='CASE_NUMBER', right_on='CASE_NUMBER')
    df_full_cair_list = df_full_cair_list[
        ['CASE_NUMBER', 'CONCATENATED_TEXT', 'L_CRASH', 'L_FIRE', 'L_PROP_DAM', 'L_INJURY', 'L_FATALITY']]
    df_full_cair_list['CASE_NUMBER'] = df_full_cair_list['CASE_NUMBER'].str.replace("'", "")
    return df_full_cair_list

def generate_VOQ_data(todays_components):
    voq_list_file_name = "voq_list_from_python_" + todays_components + ".csv"
    voq_list_full_file_path = os.path.join(BASE_DIR, voq_list_file_name)
    text_file_name = "voq_text_" + todays_components + ".csv"
    text_full_file_path = os.path.join(BASE_DIR, text_file_name)
    df = generate_voq_list_df()
    df.to_csv(voq_list_full_file_path)
    df = pd.read_csv(voq_list_full_file_path)
    odi_no_list = df['ODI_NO']
    odi_nos = ",".join(df['ODI_NO'].astype(str))
    sql = VOQ_TEXT_SQL.replace("{IDS}", odi_nos)
    df = prepare_voq_text(sql)
    df_summary = df.copy(deep=True)
    df_summary = df_summary[['ODI_NO','VOQ_SUMMARY']]
    df_summary['VOQ_SUMMARY'] = df_summary.groupby(['ODI_NO'])['VOQ_SUMMARY'].transform(lambda x: ' '.join(x))
    # drop duplicate data
    df_summary = df_summary.drop_duplicates()

    df_comp = df.copy(deep=True)
    df_comp = df_comp[['ODI_NO', 'VOQ_COMP']]
    df_comp['VOQ_COMP'] = df_comp.groupby(['ODI_NO'])['VOQ_COMP'].transform(lambda x: ' '.join(x))
    # drop duplicate data
    df_comp = df_comp.drop_duplicates()

    final_df = pd.merge(df_summary, df_comp, how='inner', left_on='ODI_NO', right_on='ODI_NO')
    # Fire
    df_fire = df.copy(deep=True)
    df_fire = df_fire[['ODI_NO', 'X_FIRE']]
    df_fire['X_FIRE'] = df_fire.groupby(['ODI_NO'])['X_FIRE'].transform(lambda x: max(x))
    # drop duplicate data
    df_fire = df_fire.drop_duplicates()
    df_fire['X_FIRE'] = df_fire['X_FIRE'].map(lambda x: "Has Fire: Yes" if x == 1 else "Has Fire: No")
    final_df = pd.merge(final_df, df_fire, how='inner', left_on='ODI_NO', right_on='ODI_NO')

    # Crash
    df_crash = df.copy(deep=True)
    df_crash = df_crash[['ODI_NO', 'X_CRASH']]
    df_crash['X_CRASH'] = df_crash.groupby(['ODI_NO'])['X_CRASH'].transform(lambda x: max(x))
    # drop duplicate data
    df_crash = df_crash.drop_duplicates()
    df_crash['X_CRASH'] = df_crash['X_CRASH'].map(lambda x: "Has Crash: Yes" if x == 1 else "Has Crash: No")
    final_df = pd.merge(final_df, df_crash, how='inner', left_on='ODI_NO', right_on='ODI_NO')

    # Fatality
    df_fatality = df.copy(deep=True)
    df_fatality = df_fatality[['ODI_NO', 'X_FATALITY']]
    df_fatality['X_FATALITY'] = df_fatality.groupby(['ODI_NO'])['X_FATALITY'].transform(lambda x: max(x))
    # drop duplicate data
    df_fatality = df_fatality.drop_duplicates()
    df_fatality['X_FATALITY'] = df_fatality['X_FATALITY'].map(lambda x: "Has Fatality: Yes" if x == 1 else "Has Fatality: No")
    final_df = pd.merge(final_df, df_fatality, how='inner', left_on='ODI_NO', right_on='ODI_NO')

    # Injury
    df_injury = df.copy(deep=True)
    df_injury = df_injury[['ODI_NO', 'X_INJURY']]
    df_injury['X_INJURY'] = df_injury.groupby(['ODI_NO'])['X_INJURY'].transform(lambda x: max(x))
    # drop duplicate data
    df_injury = df_injury.drop_duplicates()
    df_injury['X_INJURY'] = df_injury['X_INJURY'].map(
        lambda x: "Has Injury: Yes" if x == 1 else "Has Injury: No")
    final_df = pd.merge(final_df, df_injury, how='inner', left_on='ODI_NO', right_on='ODI_NO')

    print(final_df.head(5))

    final_df.to_csv(text_full_file_path)
    df_voq_list = pd.read_csv(voq_list_full_file_path)
    df_voq_text = pd.read_csv(text_full_file_path)
    df_full_voq_list = pd.merge(df_voq_list, df_voq_text, how='inner', left_on='ODI_NO', right_on='ODI_NO')
    df_full_voq_list = df_full_voq_list[
        ['ODI_NO', 'VOQ_COMP', 'VOQ_SUMMARY','X_FIRE','X_CRASH','X_FATALITY','X_INJURY', 'L_CRASH', 'L_FIRE', 'L_PROP_DAM', 'L_INJURY', 'L_FATALITY']]

    return df_full_voq_list

if __name__ == '__main__':
    todays_components = get_current_date_components()
    # Generate CAIR Data.

    full_list_file_name = "full_cair_list_with_text_" + todays_components + ".csv"
    full_list_file_path = os.path.join(BASE_DIR, full_list_file_name)
    df_full_cair_list = generate_CAIR_data(todays_components)
    df_full_cair_list.to_csv(full_list_file_path)

    # Generate VOQ Data.
    # full_list_file_name = "full_voq_list_with_text_" + todays_components + ".csv"
    # full_list_file_path = os.path.join(BASE_DIR, full_list_file_name)
    # df_full_voq_list = generate_VOQ_data(todays_components)
    # df_full_voq_list.to_csv(full_list_file_path)
    transformed_data = df_full_cair_list.to_csv(
        (Path(args.real_data) / "real_data.csv"), index=False
    )
