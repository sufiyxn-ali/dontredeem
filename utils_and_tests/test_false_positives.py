import sys
sys.path.insert(0, 'src')
from text import text_model

print('Testing FALSE POSITIVE REDUCTION\n')
print('=' * 70)

test_cases = [
    ('Your passport expired on our system just update it to avoid processing delays', 
     'LEGITIMATE', 'Legitimacy signals override keyword alert'),
    ('Your passport has expired send it immediately or face arrest', 
     'SCAM', 'Dangerous action + threat'),
    ('Please verify your account through our app to avoid delays', 
     'LEGITIMATE', 'Legitimacy context dominates'),
    ('Your Emirates ID log in to update your profile', 
     'LEGITIMATE', 'Legitimacy action request'),
    ('Your Emirates ID send it now or be deported', 
     'SCAM', 'Critical threat + action'),
]

print('\nRESULTS:\n')
for text, expected, reason in test_cases:
    score, analysis, tokens = text_model(text)
    
    # Check if result matches expectation
    is_correct = (expected == 'SCAM' and score > 0.6) or (expected == 'LEGITIMATE' and score < 0.6)
    status = 'OK' if is_correct else 'MISS'
    
    print(f'[{status}] {expected}: {score:.0%}')
    print(f'    Text: "{text[:60]}..."')
    print(f'    Reason: {reason}')
    print(f'    Analysis: {analysis}\n')

print('=' * 70)
