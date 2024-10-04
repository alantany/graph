import random
import csv
from datetime import datetime, timedelta
import ipaddress
import streamlit as st
import neo4j
from neo4j import GraphDatabase
import pandas as pd
import io

def generate_user(user_id):
    return {
        'id': f'U{user_id:05d}',
        'name': f'User{user_id}',
        'risk_score': round(random.uniform(0, 100), 2)
    }

def generate_bank_account(account_id, user_id):
    return {
        'id': f'A{account_id:05d}',
        'user_id': f'U{user_id:05d}',
        'balance': round(random.uniform(0, 100000), 2)
    }

def generate_merchant(merchant_id):
    return {
        'id': f'M{merchant_id:05d}',
        'name': f'Merchant{merchant_id}',
        'category': random.choice(['Retail', 'Food', 'Technology', 'Travel', 'Entertainment'])
    }

def generate_device(device_id):
    return {
        'id': f'D{device_id:05d}',
        'type': random.choice(['Mobile', 'Desktop', 'Tablet'])
    }

def generate_ip_address(ip_id):
    return {
        'id': f'IP{ip_id:05d}',
        'address': str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
    }

def generate_transaction(transaction_id, user_id, merchant_id, device_id, ip_id):
    amount = random.uniform(1, 20000)  # 增加交易金额范围
    status = random.choices(['Approved', 'Declined', 'Flagged'], weights=[0.8, 0.15, 0.05])[0]
    return {
        'id': f'T{transaction_id:07d}',
        'user_id': f'U{user_id:05d}',
        'merchant_id': f'M{merchant_id:05d}',
        'device_id': f'D{device_id:05d}',
        'ip_id': f'IP{ip_id:05d}',
        'amount': round(amount, 2),
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        'status': status
    }

def generate_data(num_users=1000, num_merchants=100, num_devices=500, num_ips=1000, num_transactions=5000):
    users = [generate_user(i) for i in range(num_users)]
    bank_accounts = [generate_bank_account(i, random.randint(0, num_users-1)) for i in range(num_users*2)]
    merchants = [generate_merchant(i) for i in range(num_merchants)]
    devices = [generate_device(i) for i in range(num_devices)]
    ip_addresses = [generate_ip_address(i) for i in range(num_ips)]
    
    # 生成正常交易
    transactions = [generate_transaction(i, 
                                         random.randint(0, num_users-1),
                                         random.randint(0, num_merchants-1),
                                         random.randint(0, num_devices-1),
                                         random.randint(0, num_ips-1)) 
                    for i in range(num_transactions)]
    
    # 创建一些高风险用户群组
    high_risk_users = random.sample(range(num_users), 20)
    suspicious_merchants = random.sample(range(num_merchants), 5)
    
    # 生成高风险交易
    for i in range(500):
        user_id = random.choice(high_risk_users)
        merchant_id = random.choice(suspicious_merchants)
        device_id = random.randint(0, num_devices-1)
        ip_id = random.randint(0, num_ips-1)
        transaction = generate_transaction(num_transactions + i, user_id, merchant_id, device_id, ip_id)
        transaction['status'] = 'Flagged'
        transaction['amount'] = random.uniform(15000, 20000)
        transactions.append(transaction)
    
    # 创建一些关联交易
    for i in range(200):
        user_pair = random.sample(high_risk_users, 2)
        merchant_id = random.choice(suspicious_merchants)
        device_id = random.randint(0, num_devices-1)
        ip_id = random.randint(0, num_ips-1)
        for user_id in user_pair:
            transaction = generate_transaction(num_transactions + 500 + i, user_id, merchant_id, device_id, ip_id)
            transaction['status'] = 'Flagged'
            transaction['amount'] = random.uniform(5000, 10000)
            transactions.append(transaction)

    return users, bank_accounts, merchants, devices, ip_addresses, transactions

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    users, bank_accounts, merchants, devices, ip_addresses, transactions = generate_data()
    
    write_to_csv(users, 'users.csv')
    write_to_csv(bank_accounts, 'bank_accounts.csv')
    write_to_csv(merchants, 'merchants.csv')
    write_to_csv(devices, 'devices.csv')
    write_to_csv(ip_addresses, 'ip_addresses.csv')
    write_to_csv(transactions, 'transactions.csv')

    print(f"Data generation complete. CSV files have been created.")
    print(f"Total transactions generated: {len(transactions)}")