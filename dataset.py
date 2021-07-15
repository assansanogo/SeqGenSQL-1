import argparse
import glob
import os
import json
import time
import logging
import random
import re
import copy
from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

######################################################################
## WikiSql Dataset
######################################################################

class WikiSqlDataset(Dataset):
  def __init__(self, 
               tokenizer, 
               data_dir, 
               dataset_type, 
               include_data_type = True, 
               include_sample_data = 0, 
               data_augmentation = [], 
               generated_data = [], 
               generated_data_dropout = True,
               max_input_len=512, 
               max_output_len = 200,
               include_question = False):
    
    
    # train or test
    self.dataset_type = dataset_type
    
    # full data file: path based on the data dir, train|test, and extension
    self.data_file = os.path.join(data_dir, dataset_type +'.jsonl')
    
    # full table file: path based on the data dir, train|test, and extension
    self.table_file = os.path.join(data_dir, dataset_type + '.tables.jsonl')
    
    # full questions files
    self.generated_data = generated_data

    
    self.max_input_len = max_input_len
    self.max_output_len = max_output_len
    self.data_augmentation = data_augmentation 
    self.tokenizer = tokenizer
    self.tokenizer.sep_token = '<sep>'
    self.generated_data_dropout = generated_data_dropout
    self.include_question = include_question

    #
    self.inputs = []
    self.targets = []
    self.generated_data_flag=[]

    # raw data
    self.data = []
    self.tables = {}

    # feature engineering
    self.include_data_type = include_data_type
    self.include_sample_data = include_sample_data

    # process data
    self.input_string = []
    self.target_string = []

    # gated layer
    self.gate_masks = []

    # operations    
    self.cond_ops = ['=', '>', '<', 'OP']
    self.agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    # build dataset
    self._build()
  
  def __len__(self):
    '''
    returns the length of  the raw dataset
    '''
    return len(self.data)

  def _get_encode_length(self, text):
    '''
    returns the length of encoded text
    '''
    
    return len(self.tokenizer.encode(text))

  
  
  def _build_input_output(self, raw_data):
    '''
    function which returns: 
    input, sql_statement , gate_mask  , question_pos 
    raw_data : json
    '''
    
    question, sql, table_id = raw_data['question'],raw_data['sql'], raw_data['table_id']
    columns, columns_types = self.tables[raw_data['table_id']]['header'], self.tables[raw_data['table_id']]['types']

    # the final structure is as following
    # input = question + table id + (column names, column types)
    
    
    # input = question + <eos> + table id
    input = question + self.tokenizer.sep_token + table_id
    
    # length of tokenized question
    question_pos = len(self.tokenizer.encode(question))

    # mask for gated layer
    gate_mask = [1] * self._get_encode_length(input)  # consider questions

    # Get sample data
    if self.include_sample_data > 0:
        
        # selected number of rows  will be the minimum(between the total number of rows and number of sample data)
        selected_num_rows = min(self.include_sample_data, len(self.tables[table_id]['rows']))
        #rng = np.random.default_rng()
        #row_indexes = np.sort(rng.choice(selected_num_rows + 1, size=self.include_sample_data , replace = False))
        selected_rows = self.tables[table_id]['rows'][:selected_num_rows]
   
    #add columns names + [SEP] + column data type + [SEP]
    
    for (ci, (c, ct)) in enumerate(zip(columns, columns_types)) :
        input += self.tokenizer.sep_token + c 
        gate_mask += [1] * self._get_encode_length(self.tokenizer.sep_token + c)   # consider column headers
        if self.include_data_type:
            input += self.tokenizer.sep_token + ct
            gate_mask += [0] * self._get_encode_length(self.tokenizer.sep_token + ct)  # do not use data types
        if self.include_sample_data > 0 :
            for r in selected_rows:
                input += self.tokenizer.sep_token + str(r[ci])
                gate_mask += [0] * self._get_encode_length(self.tokenizer.sep_token + str(r[ci]))  # do not use data types

    # add to the end of the input
    input += self.tokenizer.eos_token
    
    # convert the input to lowercase
    input = input.lower()
    
    # generate label - sql statement
    # SELECT 'MAX(...)', 'MIN(...)', 'COUNT(...)', 'SUM(...)', 'AVG(...)'
    sql_statement = 'SELECT ' + self.agg_ops[sql['agg']] 
    
    if sql['agg'] > 0:
        # create statement with parenthesis if there is an aggregation method  
        sql_statement += '([' +  columns[sql['sel']] + ']) FROM [' + table_id +"] "
    else:
        # create statement with parenthesis if there is no aggregation method 
        sql_statement += ' [' +  columns[sql['sel']] + '] FROM [' + table_id +"] "

    if len(sql['conds']) > 0:
        # if there is a condition WHERE
        sql_statement += 'WHERE '
        
        for c in sql['conds']:
            # in case of a condition (c is a 3-tuple)
            # append to sql statement : [ column ] = 
            
            sql_statement += '[' + columns[c[0]] + '] ' + self.cond_ops[c[1]]
            
            # if the 3rd element of c is of type: float 
            # it is the value evaluated by the condition
            
            if isinstance(c[2], (int, float)):
                sql_statement += " " + str(c[2])
                
            # if the 3rd element of c is of type: string
            # it is the value evaluated by the condition   
                
            else:
                sql_statement += " '" + c[2] + "'"
            
            # append "AND" until the last condition
            sql_statement += " AND "
        
        # remove the final "AND" as there is no statement after
        # the space will be kept before appending the <eos> token
        sql_statement = sql_statement[:-4]
     
    # append the <eos> token at the end of the sql statement
    # the sql statement is set to lowercase
    sql_statement += self.tokenizer.eos_token
    sql_statement = sql_statement.lower()

    # pad gate_mask
    # if the gate_mask is smaller than the input_length
    if len(gate_mask) < self.max_input_len:
      gate_mask += [0] * (self.max_input_len - len(gate_mask))
    
    # pad gate_mask
    # if the gate_mask is bigger than the input_length
    # the gate_mask size is limited to the size of the input
    
    elif len(gate_mask) > self.max_input_len:
      gate_mask = gate_mask[:self.max_input_len]
    
    # transform the gate_mask to the Tensor format
    gate_mask = torch.Tensor(gate_mask)
    
    # representation of the input : 
    # representation of the sql statement
    # representation of the gate_mask
    # representation of the question_pos: length of tokenized question
    return input, sql_statement , gate_mask  , question_pos  


  
  
  
  
  # This is a data augmentation method: to replace select column using randomly selected column
  def _replace_select_col(self, data):
    '''
    
    '''
    
    question = data['question'].lower()
    table_id = data['table_id']
    headers = self.tables[table_id]['header']
    
    sel_col = data['sql']['sel']
    
    sel_name = headers[sel_col].lower()

    new_col = np.random.randint(len(headers))
    new_colname = headers[new_col].lower()
    
    new_data = data.copy()
    new_data['question'] = question.replace(sel_name, new_colname)
    
    if question.find(sel_name) > -1:
        new_data['sql']['sel'] = new_col
    input, sql_statement , gate_mask, question_pos = self._build_input_output(new_data)
    return input, sql_statement, gate_mask, question_pos 

  
  
  
  
  # this is augmentation method 2: replace where value with any value from database
  def _replace_where_val(self, data):
    
    # raw data: question converted to lowercase
    question = data['question'].lower()
    
    # raw data: table id
    table_id = data['table_id']
    
    # raw data: headers from tables
    headers = self.tables[table_id]['header']

    # copy raw data
    new_data = copy.deepcopy(data)
    
    #1: Randomly pick one condition
    # number of conditions
    condition_len = len(data['sql']['conds'])
    
    # if there is more than 1 condition:
    if condition_len > 0:
        
        # choose randomly an integer to pick the modified condition index
        cond_idx = np.random.randint(condition_len, size = 1)[0]
        
        # select the condition to modify (= tuple : element, operator, element)
        cond_where = data['sql']['conds'][cond_idx]
        
        # the first element of the condition
        where_col = cond_where[0]

        # construct real sql to get all available values (old comment)
        
        # pick randomly a row_idx: new_row_idx
        new_row_idx = np.random.randint(len(self.tables[data['table_id']]['rows']))
        
        # pick the value corresponding to the new_row_idx
        new_where_value = self.tables[data['table_id']]['rows'][new_row_idx][where_col]

        # old value 
        old_where_value = data['sql']['conds'][cond_idx][2]
        
        # replace the old values by the new values
        new_data['question'] = question.replace(str(old_where_value).lower(), str(new_where_value).lower())
        new_data['sql']['conds'][cond_idx][2] = new_where_value
    
    # Generate tokens
    input, sql_statement, gate_mask, question_pos  = self._build_input_output(new_data)
    return input, sql_statement, gate_mask, question_pos 

  
  
  
  
  
  
  def __getitem__(self, index):
    # Augmenting train set only
    
    # no augmentation
    if self.dataset_type != 'train' or self.data_augmentation == []:
        input_string  = self.input_string[index]
        target_string = self.target_string[index]

        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        gate_mask = self.gate_masks[index].squeeze()

        # Input token drop out
        if self.dataset_type == 'train' and self.generated_data_flag[index] > 0 and self.generated_data_dropout:
            #drop out 1 token
            pos = np.random.randint(self.generated_data_flag[index])
            source_ids[pos] = self.tokenizer.pad_token_id
    else:
        # augmentation
        # generate input and output
        # the types of data_augmentation is based on : ['select_column', 'where_value']
        
        aug = np.random.choice(self.data_augmentation)
        
        # if the type of augmentation is : 'select_column'
        if aug == 'select_column':
            input_string, target_string, gate_mask, question_pos = self._replace_select_col(self.data[index]) 
        # if the type of augmentation is : 'where_value'   
        elif aug == 'where_value' :
            input_string, target_string, gate_mask, question_pos = self._replace_where_val(self.data[index]) 
        
        self.input_string.append(input_string)
        self.target_string.append(target_string)
        self.gate_masks.append(gate_mask)
        
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_string], max_length=self.max_input_len, pad_to_max_length=True, return_tensors="pt"
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target_string], max_length=self.max_output_len, pad_to_max_length=True, return_tensors="pt"
        )
        
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask    = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

    if self.include_question : #include the question in the dataset
      return {
        "source_ids": source_ids, 
        "target_ids": target_ids, 
        "question": self.data[index]['question'],
        "source_mask": src_mask, 
        "target_mask": target_mask, 
        "gate_mask": gate_mask}
    else:
      return {
        "source_ids": source_ids, 
        "target_ids": target_ids, 
        #"input_string": input_string, "target_string": target_string, 
        "source_mask": src_mask,
        "target_mask": target_mask,
        "gate_mask": gate_mask}

  
  def _build(self):  
    '''
    build the dataset for WikiSQL
    '''
    # load all data from table_file
    with open(self.table_file) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            self.tables[t1['id']] = t1
     
    # load all data from raw file (json object for each line)
    with open(self.data_file) as f:
        print("Loading {} ...".format(self.data_file), end="")   
        
        for idx, line in enumerate(f):
            # put into the empty list named data:
            # each row is parsed as a json
            sql = json.loads(line.strip())
            self.data.append(sql)  

            if self.dataset_type != 'train' or self.data_augmentation == []:
                # generate input and output
                input_string, sql_statement,gate_mask, _ = self._build_input_output(sql) 
                
                # append the input string into empty list of inputs strings
                # append the sql statement string into empty list of target strings
                self.input_string.append(input_string)
                self.target_string.append(sql_statement)
                
                # tokenize inputs
                # batch_encode_plus: Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
                
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input_string], max_length=self.max_input_len, pad_to_max_length=True, return_tensors="pt"
                )
                # tokenize targets
                # batch_encode_plus: Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [sql_statement], max_length=self.max_output_len, pad_to_max_length=True, return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                self.gate_masks.append(gate_mask)
                
                # for data different from training data
                # flag data as not generated (because it is not training data)
                
                self.generated_data_flag.append(0)  # not generated data
        print("Done!")    
        
    if self.dataset_type == 'train':
        # the generated files (all json files) 
        
        for gen_file in self.generated_data:
            print("Loading {} ...".format(gen_file), end="")
            with open(gen_file) as f:
              
                for idx, line in enumerate(f):
                    # sql load each line as a json object
                    sql = json.loads(line.strip())
                    
                    # raw dataset: append each line (json object)
                    self.data.append(sql)  

                    # in case of no data augmentation
                    # process the lines to parse the json into inputs/outputs
                    
                    if self.data_augmentation == []:
                        
                        # generate input and output
                        input_string, sql_statement, gate_mask,question_pos = self._build_input_output(sql) 

                        self.input_string.append(input_string)
                        self.target_string.append(sql_statement)
                       
                        # tokenize inputs
                        tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [input_string], max_length=self.max_input_len, pad_to_max_length=True, return_tensors="pt"
                        )
                        # tokenize targets
                        tokenized_targets = self.tokenizer.batch_encode_plus(
                            [sql_statement], max_length=self.max_output_len, pad_to_max_length=True, return_tensors="pt"
                        )

                        self.inputs.append(tokenized_inputs)
                        self.targets.append(tokenized_targets)
                        self.gate_masks.append(gate_mask)
                        self.generated_data_flag.append(question_pos)  # generated data
            print("Done!")

