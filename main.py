import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from tensorflow.python.platform import gfile
import tensorflow as tf
import random

def Generate_Demand_Dataset():
    # index 생성
    index = pd.date_range(start='1/1/2013', end='7/31/2020')
    data = np.random.randint(1, high=100, size=len(index))

    # Raw Data 불러오기
    cd = os.getcwd()
    file_name = '한국전력거래소_시간별 전력수요량_20200731.csv'
    file_path = os.path.join(cd, file_name)
    df = pd.read_csv(file_name, encoding='cp949')
    df['날짜'] = pd.to_datetime(df['날짜'])

    # 일별 거래량 합치기
    for idx, i in enumerate(index):
        tdf = df[df['날짜'] == i]
        tdf.pop('날짜')
        tdf = tdf.transpose()
        print(tdf)
        data[idx] = tdf.sum()

    # Demand_Dataset 생성
    d_df = pd.DataFrame({'Index': index, 'Demand': data})
    d_df = d_df.set_index('Index')
    d_df.to_csv(os.path.join(cd, 'Demand_Data.csv'))

def Demand_Dataset():
    # Demand_dataset 불러오기
    cd = os.getcwd()
    file_name = 'Demand_Data.csv'
    file_path = os.path.join(cd, file_name)
    df = pd.read_csv(file_name, encoding='cp949')
    df['Index'] = pd.to_datetime(df['Index'])
    df = df.set_index('Index')
    # 연도, 월, 요일 컬럼 추가하기
    Year = []
    Month = []
    Weekday = []
    for i in df.index:
        Year.append(i.year)
        Month.append(i.month)
        Weekday.append(i.weekday())
    df['Year'] = Year
    df['Month'] = Month
    df['Weekday'] = Weekday
    df['Demand']=df['Demand']/1000
    return df

def plot_on_month(df):
    monthly_demand = []
    for i in range(1, 13):
        temp = df.loc[df['Month'] == i]
        temp = temp['Demand'].mean()
        monthly_demand.append(temp)
    plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], monthly_demand)
    plt.xlabel('Month')
    plt.ylabel('Average Daily Demand')
    plt.ylim(bottom=0)
    plt.title("Average Daily Demand per Month (20130101~20200731)")
    plt.show()

def plot_on_weekday(df):
    weekdayly_demand = []
    for i in range(0, 7):
        temp = df.loc[df['Weekday'] == i]
        temp = temp['Demand'].mean()
        weekdayly_demand.append(temp)
    plt.bar(['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], weekdayly_demand)
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Daily Demand')
    plt.ylim(bottom=0)
    plt.title("Average Daily Demand per Day of the Week (20130101~20200731)")
    plt.show()

def plot_on_year(df):
    yearly_demand = []
    for i in range(2013, 2020):
        temp = df.loc[df['Year'] == i]
        temp = temp['Demand'].mean()
        yearly_demand.append(temp)
    plt.bar(['2013', '2014', '2015', '2016', '2017', '2018', '2019'], yearly_demand)
    plt.xlabel('Year')
    plt.ylabel('Average Daily Demand')
    plt.ylim(bottom=0)
    plt.title("Average Daily Demand per Year (20130101~20101231)")
    plt.show()

def s_Q_policy(train_data, test_data):
    # 일일 수요의 평균 및 표준편차
    mean = train_data['Demand'].mean()
    std = train_data['Demand'].std()
    print('일평균 수요 평균 및 표준편차', mean, std)
    A = 20000 # Setup Cost 2만원
    D = 365*mean #수요 per Year
    r = 0.2 #연평균 재고비용
    v = 1000 # 물품 가격 1000원
    L = 1 # Leadtime 1일
    # EOQ 계산
    Q = round((2*A*D/(r*v))**0.5)
    # SS 계산
    T = Q/D*365
    B2 = 0.25
    a = Q*r/(D*B2)
    k = 2.156860196
    s = math.ceil(mean+k*std)
    print('2013년~2018년까지의 데이터로부터 분포 추정')
    print('(', s, ',', Q, ') Policy(s, Q)')

    Inventory = s + Q
    Backorder = 0
    Delivery = [0, 0]
    OH_Inventory = Inventory + Delivery[0] +Delivery[1] - Backorder
    Inventory_list=[]
    Backorder_list=[]
    Sales_list=[]
    Order_list=[]
    for i in test_data['Demand']:
        # 배송 받음
        Inventory += Delivery[0]
        Delivery[0] = Delivery[1]
        Delivery[1] = 0
        # 판매량 계산
        Sales = min(Inventory, round(i) + Backorder)
        Sales_list.append(Sales)
        # 수요만큼 재고 소진
        Inventory -= round(i) + Backorder
        # Backorder 발생
        Backorder = max(0, -Inventory)
        Inventory = max(Inventory, 0)
        Inventory_list.append(Inventory)
        Backorder_list.append(Backorder)
        # On Hand Inventory 계산
        OH_Inventory = Inventory + Delivery[0] +Delivery[1] - Backorder
        # s 도달 시 주문
        if OH_Inventory <= s:
            Delivery[1] = Q
            Order_list.append(1)
        else:
            Order_list.append(0)
    test_data['Inventory'] = Inventory_list
    test_data['Backorder'] = Backorder_list
    test_data['Sales'] = Sales_list
    test_data['Order'] = Order_list
    test_result = test_data
    return test_result

def analyze(df):
    # Order Quantity
    Q = 10317
    # Setup cost
    A = 19657
    # variable cost
    v = 1000
    # Inventory cost
    r = 0.2
    # Backorder cost
    B2 = 0.25
    # Sales revenue
    p = 1200
    # total profit
    pi = -df['Order'].sum()*(A+Q*v)-df['Inventory'].sum()*0.2/365*v-df['Backorder'].sum()*(B2/365)*v+df['Sales'].sum()*p
    # Average Inventory
    AI = df['Inventory'].mean()
    # Average Backorder
    AB = df['Backorder'].mean()
    # Backorder 발생 비율
    BP = len(df.loc[df['Backorder'] > 0])/len(df)
    # 총 판매량
    TS = df['Sales'].sum()
    print('총수익', pi, '평균재고', AI, '평균 Backorder', AB, 'Backorder 발생 확률', BP, '총판매량', TS)
    print('주문비용 재고비용 Backorder비용, 판매수익',-df['Order'].sum()*(A+Q*v),-df['Inventory'].sum()*0.2/365*v,-df['Backorder'].sum()*(B2/365)*v, df['Sales'].sum()*p)
    pi = -df['Inventory'].sum()*0.2/365*v -df['Backorder'].sum()*(B2/365)*v
    return(pi)

def train_RL(train_data):
    # Given Variable
    # Order Quantity
    Q = 10317
    # Setup cost
    A = 19657
    # variable cost
    v = 1000
    # Inventory cost
    r = 0.2
    # Backorder cost
    B2 = 0.25
    # Sales revenue
    p = 1200
    ###################################################
    # model hyperparameter
    epsilon = 0.01
    gamma = 0.95
    lr = 0.001
    num_epoch = 20
    ###################################################
    Inventory = 12109
    Backorder = 0
    Delivery = [0, 0]
    OH_Inventory = Inventory-Backorder+Delivery[0]+Delivery[1]
    Year = 2013
    Month = 1
    ###################################################
    actions = ['order', 'stay']
    state = [OH_Inventory, Year, Month, Backorder]
    input_dim = len(state)
    policy = QLearningDecisionPolicy(epsilon=epsilon, gamma=gamma, lr=lr, actions=actions, input_dim=input_dim, model_dir="model")
    run_simulations(df=train_data, policy=policy, Inventory=Inventory, Backorder=Backorder, Delivery=Delivery, OH_Inventory=OH_Inventory, Year=Year, Month=Month, num_epoch=num_epoch)
    policy.save_model("Graduate_Project")

class QLearningDecisionPolicy:
    def __init__(self, epsilon, gamma, lr, actions, input_dim, model_dir):
        # select action function hyperparameter
        self.epsilon = epsilon
        # q functions hyperparameter
        self.gamma = gamma
        # neural network hyperparmeter
        self.lr = lr
        # actions
        self.actions = actions
        output_dim = len(actions)

        # neural network input and output placeholder
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [output_dim])

        # TODO: build your Q-network
        # 2-layer fully connected network
        fc = tf.compat.v1.layers.dense(self.x, 20, activation=tf.nn.relu)
        self.q = tf.compat.v1.layers.dense(fc, output_dim)

        # loss
        loss = tf.square(self.y - self.q)

        # train operation
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        # session
        self.sess = tf.compat.v1.Session()

        # initalize variables
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # saver
        self.saver = tf.compat.v1.train.Saver()

        # restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("load model: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def select_action(self, current_state, is_training=True):

        if random.random() >= self.epsilon or not is_training:
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]
        else:  # randomly select action
            action = self.actions[random.randint(0, len(self.actions)-1)]
        return action

    def update_q(self, current_state, action, reward, next_state):
        # Q(s, a)
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
        # Q(s', a')
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        # a' index
        next_action_idx = np.argmax(next_action_q_vals)
        # create target
        action_q_vals[0, self.actions.index(action)] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        # delete minibatch dimension
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: current_state, self.y: action_q_vals})

    def save_model(self, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        checkpoint_path = output_dir + '/model'
        self.saver.save(self.sess, checkpoint_path)

def do_action(action_list, action, Inventory, Backorder, Delivery, OH_Inventory, Demand):
    # EOQ 주문량
    Q = 10317
    # 배송 받음
    Inventory += Delivery[0]
    Delivery[0] = Delivery[1]
    Delivery[1] = 0
    # 판매량 계산
    Sales = min(Inventory, round(Demand) + Backorder)
    # 수요만큼 재고 소진
    Inventory -= round(Demand) + Backorder
    # Backorder 발생
    Backorder = max(0, -Inventory)
    Inventory = max(Inventory, 0)
    # On Hand Inventory 계산
    OH_Inventory = Inventory + Delivery[0] + Delivery[1] - Backorder
    # s 도달 시 주문
    if action == 'order':
        Delivery[1] = Q
        Order = 1
    else:
        Delivery[1] = 0
        Order = 0
    return Inventory, Backorder, Delivery, OH_Inventory, Sales, Order

def run_simulation(policy, initial_Inventory, initial_Backorder, initial_Delivery, initial_OH_Inventory, df):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    Inventory = initial_Inventory
    Backorder = initial_Backorder
    Delivery = initial_Delivery
    OH_Inventory = initial_OH_Inventory
    Inventory_list = []
    Backorder_list = []
    Order_list = []
    Sales_list = []
    AI_next = 0
    AB_next = 0
    k = -1
    for idx, row in df.iterrows():
        k+=1
        ##### TODO: define current state
        current_state = np.asmatrix(np.hstack(([OH_Inventory], [row['Year']], [row['Month'],], [Backorder])))
        Demand = row['Demand']
        AI = AI_next
        AB = AB_next

        ##### select action & update portfolio values
        action = policy.select_action(current_state, True)
        action_seq.append(action)
        action_count[policy.actions.index(action)] += 1
        Inventory, Backorder, Delivery, OH_Inventory, Sales, Order = do_action(policy.actions, action, Inventory, Backorder, Delivery, OH_Inventory, Demand)

        ##### list append
        Inventory_list.append(Inventory)
        Backorder_list.append(Backorder)
        Sales_list.append(Sales)
        Order_list.append(Order)

        AI_next = sum(Inventory_list)/len(Inventory_list)
        AB_next = sum(Backorder_list)/len(Backorder_list)
        if k >=30:
            AI_next = sum(Inventory_list[-30:]) / len(Inventory_list[-30:])
            AB_next = sum(Backorder_list[-30:]) / len(Backorder_list[-30:])


        ##### TODO: define reward
        # calculate reward from taking an action at a state
        reward = -(AI_next)-Backorder

        ##### TODO: define next state
        next_state = np.asmatrix(np.hstack(([OH_Inventory], [row['Year']], [row['Month']], [Backorder])))
        ##### update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final profit
    df['Inventory'] = Inventory_list
    df['Backorder'] = Backorder_list
    df['Sales'] = Sales_list
    df['Order'] = Order_list
    profit = analyze(df)
    return profit, action_count, np.asarray(action_seq)

def run_simulations(df, policy, Inventory, Backorder, Delivery, OH_Inventory, Year, Month, num_epoch):
    PRINT_EPOCH = num_epoch
    best_profit = -100000000000000
    # final_portfolios = list()
    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        profit, action_count, action_seq = run_simulation(policy, Inventory, Backorder, Delivery, OH_Inventory, df)
        # final_portfolios.append(final_portfolio)
        print('actions : ', *zip(policy.actions, action_count), )

        # if (epoch + 1) % PRINT_EPOCH == 0:
        #     action_seq2 = np.concatenate([['.'], action_seq[:-1]])
        #     for i, a in enumerate(policy.actions[:-1]):
        #         plt.figure(figsize=(40, 20))
        #         plt.title('Company {} / Epoch {}'.format(a, epoch + 1))
        #         plt.plot(open_prices[0: len(action_seq),i], 'grey')
        #         plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=1) # sell
        #         plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=1)  # buy
        #         plt.show()

        ##### save if best portfolio value is updated
        if best_profit < profit:
            best_profit = profit
            policy.save_model("Best_Model")

    # print(final_portfolios[-1])

def test_RL(test_data):
    actions = ['order', 'stay']
    input_dim = 4
    policy1 = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, "Best_Model")
    Inventory = 12109
    run_simulations(df=test_data, policy=policy1, Inventory=Inventory, Backorder=0, Delivery=[0, 0], OH_Inventory=0, Year=2019, Month=1, num_epoch=1)

Demand_Data = Demand_Dataset()
# 전력 수요 분석 그래프
# plot_on_month(Demand_Data)
# plot_on_weekday(Demand_Data)
# plot_on_year(Demand_Data)

train_data = Demand_Data.loc[Demand_Data['Year'] < 2019]
test_data = Demand_Data.loc[Demand_Data['Year'] >= 2019]
# test_result = s_Q_policy(train_data, test_data)
# analyze(test_result)
# train_RL(train_data)
test_RL(test_data)

