from .BaseEnv import BaseEnv
import tiktoken
import logging, json, random

def get_recall(res, std):
    if res == None:
        return 0
    # print(std)
    res = list(set(res))
    std_set = set(std)
    ct = 0
    for step_id in res:
        if step_id in std:
            ct += 1
    return ct/len(std_set)


class MemBenchEnv(BaseEnv):

    INITIAL_INSTRUACTION = 'Please help me record the following information. If there are any questions within the information, please help me answer them.'

    def __init__(self, config, path_i):
        super().__init__(config)

        self.dataset = self.load_dataset(path_i)
        self.token_count = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.reset(0)

    def load_dataset(self, path_i):
        all_data = load_json(self.config['dataset_path'][path_i])
        print(self.config['dataset_path'])
        test_data = []
        for question_type, traj_list_all in all_data.items():
            if question_type in self.config['dataset_type']:
                for _, traj_list in traj_list_all.items():
                    for traj in traj_list:
                        test_data.append(traj)
        return test_data

    def reset(self, traj_i):
        """
        Reset the environment.
        """
        self.current_step = 0 
        self.token_count = 0
        self.task_info = self.dataset[traj_i]  # 取第i个traj
        logging.info("Task has been reset successully: {}.".format(traj_i))
        return {'message': self.INITIAL_INSTRUACTION}, 0, False, {'signal':'Success.', 'step_id': self.current_step}

    def step(self, action, mode):
        """
        Transform (action -> observation, reward, terminated, info).
        """
        # if mode == 'ThirdAgent':
        self.current_step += 1
        if self.current_step - 1 < len(self.task_info['message_list']):
            # User will just give a statement of her/his information.
            # print(self.task_info['message_list'][self.current_step-1])
            return {'message': self.task_info['message_list'][self.current_step-1]}, 0, False, {'signal':'Success.', 'step_id': self.current_step}, None
        elif self.current_step - 1 == len(self.task_info['message_list']):
            # User will ask a question at the end of trajectory.
            return {
                'question': self.task_info['QA']['question'],
                'time': self.task_info['QA']['time'],
                'choices': self.task_info['QA']['choices']
            }, 0, False, {'signal':'Success.', 'step_id': self.current_step}, None  # 不进行recall
        elif self.current_step - 1 == len(self.task_info['message_list']) + 1:
            # User will judge whether the predicted answer is correct for the question.
            predict, answer = self.task_info['QA']['ground_truth'], action['response']
            correct = action['response'] == self.task_info['QA']['ground_truth']

            recall = None
            if 'memory_index' in action:
                recall = get_recall(action['memory_index'], self.task_info['QA']['target_step_id'])
            
            logging.info("Comparing %s(GT) with %s(ANS)" % (predict, answer))
            if correct:
                return {'message':'Answer correct!'}, 1, True, {'signal':'Success.', 'step_id': self.current_step}, recall
            else:
                return {'message':'Answer wrong!'}, 0, True, {'signal':'Success.', 'step_id': self.current_step}, recall
    
    def step_cap(self, action, mode):
        self.current_step += 1
        if  self.current_step - 1 < self.task_info['QA']['target_step_id'][-1]: 
            
            if isinstance(self.task_info['message_list'][self.current_step-1], dict):
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['user'])) + len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['agent']))
            else:
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]))

            return {'message': self.task_info['message_list'][self.current_step-1]}, 0, False, {'signal':'Success.', 'step_id': self.current_step}, False  # 用最后一位标识是否开始了容量测量
        elif self.current_step - 1 == self.task_info['QA']['target_step_id'][-1]:  # 这一步要给出问题，不能回答
            if isinstance(self.task_info['message_list'][self.current_step-1], dict):
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['user'])) + len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['agent']))
            else:
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]))

            return {
                'message': self.task_info['message_list'][self.current_step-1],
                'question': self.task_info['QA']['question'],
                'time': self.task_info['QA']['time'],
                'choices': self.task_info['QA']['choices']
            }, 0, False, {'signal':'Success.', 'step_id': self.current_step}, False
        
        elif self.current_step - 1 < len(self.task_info['message_list']):
            predict, answer = self.task_info['QA']['ground_truth'], action['response']
            correct = action['response'] == self.task_info['QA']['ground_truth']

            if correct:
                correct_ = (self.token_count, 1)
            else:
                correct_ = (self.token_count, 0)
            
            # 传过去的下一个message不在测量范围内
            if isinstance(self.task_info['message_list'][self.current_step-1], dict):
                # print(self.task_info['message_list'][self.current_step-1])
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['user'])) + len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]['agent']))
            else:
                self.token_count += len(self.encoding.encode(self.task_info['message_list'][self.current_step-1]))

            return {
                'message': self.task_info['message_list'][self.current_step-1],
                'question': self.task_info['QA']['question'],
                'time': self.task_info['QA']['time'],
                'choices': self.task_info['QA']['choices']
            }, correct_, False, {'signal':'Success.', 'step_id': self.current_step}, True
        
        elif self.current_step - 1 == len(self.task_info['message_list']):
            predict, answer = self.task_info['QA']['ground_truth'], action['response']
            correct = action['response'] == self.task_info['QA']['ground_truth']
            if correct:
                return {'message':'Answer correct!'}, (self.token_count, 1), True, {'signal':'Success.', 'step_id': self.current_step}, True
            else:
                return {'message':'Answer wrong!'}, (self.token_count, 0), True, {'signal':'Success.', 'step_id': self.current_step}, True
    

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data