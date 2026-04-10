
creating an attack flag

```
# Binary classification target
# Maps normal traffic to 0 and any type of attack to 1
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
```

### Creating multi-classs classification target

A custom function `map_attack` checks the type of attack and assigns it an integer:

- `0` for normal traffic
- `1` for DoS attacks
- `2` for Probe attacks
- `3` for Privilege Escalation attacks
- `4` for Access attacks

```
# Multi-class classification target categories
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
               'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                     'rootkit', 'sqlattack', 'xterm']
access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                  'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                  'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                  'xclock', 'xsnoop']

def map_attack(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0

# Assign multi-class category to each row
df['attack_map'] = df['attack'].apply(map_attack)
```


### Encoding Categorical Variables


protocol type and service arent numeric values so cant be processed by ML algorithms. For that we use encoding.

one-hot-encoding via get_dummies in pandas

`# Encoding categorical variables features_to_encode = ['protocol_type', 'service'] encoded = pd.get_dummies(df[features_to_encode])`



## Seleting Numerical Features


Beyond categorical variables, the dataset contains a range of numeric features that describe various aspects of network traffic. These include basic metrics like `duration`, `src_bytes`, and `dst_bytes`, as well as more specialized features such as `serror_rate` and `dst_host_srv_diff_host_rate`, which capture statistical properties of the network sessions. By selecting these numeric features, we ensure the model has access to both raw volume data and more nuanced, derived statistics that help distinguish normal from abnormal patterns.

![[Pasted image 20260410114404.png]]


## Combining numerical and categorical Features

        python
`# Combine encoded categorical variables and numeric features train_set = encoded.join(df[numeric_features]) # Multi-class target variable multi_y = df['attack_map']`





## Splitting the Dataset

### Splitting Data into Training and Test Sets

We use `train_test_split` to allocate a portion of the data for testing, ensuring that our final evaluations occur on unseen data.

        python
`# Split data into training and test sets for multi-class classification train_X, test_X, train_y, test_y = train_test_split(train_set, multi_y, test_size=0.2, random_state=1337)`

### Creating a Validation Set from the Training Data

We further split the training data to create a validation set. This supports model tuning and hyperparameter optimization without contaminating the final test data.

        python
`# Further split the training set into separate training and validation sets multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1337)`

