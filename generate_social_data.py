import random
import csv
from datetime import datetime, timedelta
import os

def generate_social_user(user_id):
    return {
        'id': f'SU{user_id:05d}',
        'name': f'SocialUser{user_id}',
        'followers_count': random.randint(0, 1000),
        'following_count': random.randint(0, 100),
        'account_creation_date': (datetime.now() - timedelta(days=random.randint(0, 3650))).isoformat(),
        'is_verified': random.choice([True, False]),
        'activity_score': random.uniform(0, 100)
    }

def generate_post(post_id, user_id):
    return {
        'id': f'P{post_id:07d}',
        'user_id': f'SU{user_id:05d}',
        'content': f'This is post {post_id} by user {user_id}',
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        'likes_count': random.randint(0, 100),
        'shares_count': random.randint(0, 10)
    }

def generate_follow_relationship(user1_id, user2_id):
    return {
        'follower_id': f'SU{user1_id:05d}',
        'following_id': f'SU{user2_id:05d}',
        'follow_date': (datetime.now() - timedelta(days=random.randint(0, 1000))).isoformat()
    }

def generate_interest(interest_id):
    interests = ['Technology', 'Sports', 'Music', 'Travel', 'Food', 'Fashion', 'Art', 'Politics', 'Science', 'Health']
    return {
        'id': f'I{interest_id:03d}',
        'name': interests[interest_id % len(interests)]
    }

def generate_user_interest(user_id, interest_id):
    return {
        'user_id': f'SU{user_id:05d}',
        'interest_id': f'I{interest_id:03d}',
        'affinity_score': random.uniform(0, 1)
    }

def generate_social_data(num_users=1000, num_posts=5000, num_interests=10):
    users = [generate_social_user(i) for i in range(num_users)]
    posts = [generate_post(i, random.randint(0, num_users-1)) for i in range(num_posts)]
    
    # Generate follow relationships (each user follows about 10% of other users)
    follow_relationships = []
    for i in range(num_users):
        for j in random.sample(range(num_users), num_users // 10):
            if i != j:
                follow_relationships.append(generate_follow_relationship(i, j))
    
    interests = [generate_interest(i) for i in range(num_interests)]
    
    # Generate user interests (each user has 1-3 interests)
    user_interests = []
    for user_id in range(num_users):
        for interest_id in random.sample(range(num_interests), random.randint(1, 3)):
            user_interests.append(generate_user_interest(user_id, interest_id))

    return users, posts, follow_relationships, interests, user_interests

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    social_dir = 'social'
    os.makedirs(social_dir, exist_ok=True)

    users, posts, follow_relationships, interests, user_interests = generate_social_data()
    
    write_to_csv(users, os.path.join(social_dir, 'social_users.csv'))
    write_to_csv(posts, os.path.join(social_dir, 'posts.csv'))
    write_to_csv(follow_relationships, os.path.join(social_dir, 'follow_relationships.csv'))
    write_to_csv(interests, os.path.join(social_dir, 'interests.csv'))
    write_to_csv(user_interests, os.path.join(social_dir, 'user_interests.csv'))

    print(f"Social data generation complete. CSV files have been created in the '{social_dir}' directory.")
    print(f"Total users: {len(users)}")
    print(f"Total posts: {len(posts)}")
    print(f"Total follow relationships: {len(follow_relationships)}")
    print(f"Total interests: {len(interests)}")
    print(f"Total user-interest relationships: {len(user_interests)}")