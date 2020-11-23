import json


def fix_issues(spider_train):
    # fix some errors in the training set
    err_ids = [6795, 6797, 6798, 6799, 6800]
    for id in err_ids:
        spider_train[id]['question'] = spider_train[id]['question'].replace('first', 'last')
    spider_train[1855]['question'] = spider_train[1855]['question'].replace('movies', 'moves')
    spider_train[1908][
        'query'] = "SELECT count(*) FROM endowment WHERE amount  >  8.5 GROUP BY school_id HAVING count(*)  >  1"
    spider_train[2013]['question'] = spider_train[2013]['question'].replace('manager_name', 'manager name')
    spider_train[2502][
        'query'] = "SELECT T2.title ,  T2.director ,  max(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID WHERE director != \"null\" GROUP BY director"
    spider_train[2503][
        'query'] = "SELECT T2.title ,   T2.director ,  max(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID WHERE director != \"null\" GROUP BY director"
    spider_train[2504][
        'query'] = "SELECT T2.title ,  T1.rID , min(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID GROUP BY T1.rID"
    spider_train[2505][
        'query'] = "SELECT T2.title ,  T1.rID , min(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID GROUP BY T1.rID"
    spider_train[2506][
        'query'] = "SELECT T2.title ,  T2.director ,  min(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID GROUP BY T2.director"
    spider_train[2507][
        'query'] = "SELECT T2.title ,  T2.director ,  min(T1.stars) FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID GROUP BY T2.director"
    spider_train[2836]['query'] = spider_train[2836]['query'].replace("HAVING", "WHERE")
    spider_train[3153]['query'] = spider_train[3153]['query'].replace(
        'JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ', '')
    spider_train[3515][
        'query'] = "SELECT first_name ,   last_name ,   department_id ,  MAX(salary) FROM employees GROUP BY department_id"
    spider_train[3516][
        'query'] = "SELECT first_name ,   last_name ,  department_id ,  MAX(salary) FROM employees GROUP BY department_id"

    # change some SQL queries as the current model does not support them
    spider_train[3673][
        'query'] = "SELECT count(*) FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_winner  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' UNION SELECT count(*) FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_loser  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings'"
    spider_train[3674][
        'query'] = "SELECT count(*) FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_winner  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings' UNION SELECT count(*) FROM postseason AS T1 JOIN team AS T2 ON T1.team_id_loser  =  T2.team_id_br WHERE T2.name  =  'Boston Red Stockings'"
    spider_train[4650]['question'] = spider_train[4650]['question'].replace('minumum', 'minimum')
    spider_train[5030][
        'query'] = "SELECT COUNT(*) FROM tryout WHERE pPos  =  \'goalie\' INTERSECT SELECT cName FROM  tryout WHERE pPos  =  \'mid\'"
    spider_train[5031][
        'query'] = 'SELECT COUNT(*) FROM tryout WHERE pPos  =  \'goalie\' INTERSECT SELECT cName FROM  tryout WHERE pPos  =  \'mid\''
    spider_train[5036][
        'query'] = "SELECT COUNT(*) FROM college AS T1 JOIN tryout AS T2 ON T1.cName  =  T2.cName WHERE T2.pPos  =  \'mid\' EXCEPT SELECT count(*) FROM college AS T1 JOIN tryout AS T2 ON T1.cName  =  T2.cName WHERE T2.pPos  =  \'goalie\'"
    spider_train[5037][
        'query'] = "SELECT COUNT(*) FROM college AS T1 JOIN tryout AS T2 ON T1.cName  =  T2.cName WHERE T2.pPos  =  \'mid\' EXCEPT SELECT count(*) FROM college AS T1 JOIN tryout AS T2 ON T1.cName  =  T2.cName WHERE T2.pPos  =  \'goalie\'"

    spider_train[5393]['question'] = spider_train[5393]['question'].replace('problems?s', 'problems?')
    spider_train[5441]['question'] = spider_train[5441]['question'].replace('Goergia', 'Georgia')
    spider_train[6092]['question'] = spider_train[6092]['question'].replace('\\', '')
    spider_train[6402]['question'] = spider_train[6402]['question'].replace('come', 'code')
    spider_train[6552]['question'] = spider_train[6552]['question'].replace('neames', 'names')
    spider_train[6556]['question'] = spider_train[6556]['question'].replace('each each', 'each year')
    spider_train[6563]['query'] = spider_train[6563]['query'].replace("HAVING", "WHERE")
    spider_train[6564]['query'] = spider_train[6564]['query'].replace("HAVING", "WHERE")
    spider_train[6853]['query'] = "SELECT count(city) FROM airports GROUP BY city HAVING count(*)  >  3"
    spider_train[6854]['query'] = "SELECT count(city) FROM airports GROUP BY city HAVING count(*)  >  3"
    spider_train[6908]['question'] = spider_train[6908]['question'].replace('??', '?')
    spider_train[6961]['question'] = spider_train[6961]['question'].replace('archtect', 'architect')


with open('data/train_spider.json') as spider, open('data/spider_ant_train.json') as spider_ant:
    spider = list(json.load(spider))
    spider_ant = list(json.load(spider_ant))
    ret = []
    for i, (ex, a) in enumerate(zip(spider, spider_ant)):
        cur = {'id': i, 'db_id': ex['db_id'], 'query': ex['query'], 'question': ex['question'], 'ant': a}
        ret.append(cur)
    fix_issues(ret)
    with open('data/slsql_train.json', 'wt') as output:
        json.dump(ret, output, sort_keys=True, indent=4, separators=(',', ': '))

with open('data/dev.json') as spider, open('data/spider_ant_dev.json') as spider_ant:
    spider = list(json.load(spider))
    spider_ant = list(json.load(spider_ant))
    ret = []
    for i, (ex, a) in enumerate(zip(spider, spider_ant)):
        cur = {'id': i, 'db_id': ex['db_id'], 'query': ex['query'], 'question': ex['question'], 'ant': a}
        ret.append(cur)
    with open('data/slsql_dev.json', 'wt') as output:
        json.dump(ret, output, sort_keys=True, indent=4, separators=(',', ': '))
