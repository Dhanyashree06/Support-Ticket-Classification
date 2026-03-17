"""
=============================================================
  Support Ticket Classification & Prioritization System
  Automated ML pipeline for customer support operations
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

np.random.seed(42)

ticket_templates = {
    "Billing": {
        "High": [
            "I was charged twice for my subscription this month. Please refund immediately.",
            "Unauthorized charge of $199 appeared on my account. This is fraud!",
            "My payment was declined but money was still deducted from my bank account.",
            "I cancelled my subscription but was still billed. Need urgent resolution.",
            "Double billing issue — two charges of $49.99 appeared on my statement today.",
            "Wrong amount charged. I signed up for the $9 plan but was charged $99.",
            "Payment processing error but my card was charged. Need refund ASAP.",
            "Invoice shows incorrect charges for enterprise tier I never subscribed to.",
        ],
        "Medium": [
            "Can you explain the charges on my latest invoice? Some items look unfamiliar.",
            "How do I update my billing address for future invoices?",
            "I'd like to switch from monthly to annual billing. How does proration work?",
            "My invoice is missing the VAT number. I need it for business expenses.",
            "I upgraded mid-cycle. Can you send an updated invoice showing the changes?",
            "Where can I download my past invoices for accounting purposes?",
            "I need a receipt for my last payment for reimbursement.",
        ],
        "Low": [
            "Just wondering what payment methods you accept.",
            "Do you offer student discounts?",
            "Can I get a copy of my billing history for the last year?",
            "When is the billing cycle reset each month?",
            "Do you charge in USD or can I pay in EUR?",
            "Is there a free trial available before committing to a paid plan?",
        ],
    },
    "Technical Issue": {
        "High": [
            "The entire platform is down. We cannot access anything. This is critical for our business.",
            "Data loss detected — files I uploaded yesterday are completely gone.",
            "API returning 500 errors on all endpoints. Production system is broken.",
            "Security breach — I see another user's private data in my dashboard.",
            "System crash every time I try to export data. Lost 3 hours of work.",
            "Login is completely broken. No one on our team can access the platform.",
            "Database sync failure causing data corruption across all our records.",
            "Critical bug: payments are going through but orders are not being created.",
        ],
        "Medium": [
            "The mobile app crashes intermittently when switching between tabs.",
            "Export feature is not working properly for CSV files over 10MB.",
            "Dashboard charts are not loading — showing spinner indefinitely.",
            "Search results are delayed by about 30 seconds. Very frustrating.",
            "Notifications are not being delivered to my email even though enabled.",
            "The bulk upload feature fails after about 50 records.",
            "Integration with Slack stopped working after last week's update.",
            "Two-factor authentication codes are arriving 5 minutes late.",
        ],
        "Low": [
            "Minor UI glitch — button overlaps text on mobile in portrait mode.",
            "The dark mode toggle doesn't remember my preference after refresh.",
            "Font rendering looks slightly off on my 4K monitor.",
            "Sorting by date column doesn't seem to work correctly in the table.",
            "Some icons are missing in the settings panel on Firefox browser.",
            "The loading animation seems slower than usual today.",
            "Tooltip text is cut off on the analytics page.",
        ],
    },
    "Account": {
        "High": [
            "My account has been hacked. Someone changed my password and email address.",
            "Account locked after too many login attempts. I need immediate access.",
            "I cannot access my account and have important client data inside.",
            "Suspicious login from unknown location — I think my account is compromised.",
            "Team admin accidentally deleted all user accounts. Need restoration urgently.",
        ],
        "Medium": [
            "How do I add team members to my organization account?",
            "I need to transfer my account to a different email address.",
            "Can I merge two accounts that I accidentally created?",
            "How do I change my username without losing any data or history?",
            "I need to set up SSO for my company. Can you guide me through the process?",
            "How do I downgrade my account from Enterprise to Pro plan?",
            "I want to delete my account but need to export my data first.",
            "How do I set permissions for different team roles?",
        ],
        "Low": [
            "How do I update my profile picture?",
            "Where can I change my notification preferences?",
            "How do I add my company logo to my profile?",
            "Can I change my display name in the app?",
            "How do I view my account activity history?",
            "Is there a way to set up a backup email address?",
        ],
    },
    "General Query": {
        "High": [
            "Our contract expires tomorrow and we haven't received renewal information.",
            "We need SLA documentation urgently for a compliance audit tomorrow.",
            "Legal department requires data processing agreement signed by end of day.",
        ],
        "Medium": [
            "What is the difference between the Pro and Enterprise plans?",
            "Do you offer a white-label solution for agencies?",
            "What is your data retention policy for deleted accounts?",
            "Can your API handle 10,000 requests per minute for our use case?",
            "Does your platform comply with GDPR regulations?",
            "Do you have a native integration with Salesforce?",
            "What kind of support do Enterprise customers receive?",
            "How long does onboarding typically take for a team of 50 people?",
        ],
        "Low": [
            "What are your business hours for customer support?",
            "Do you have documentation or tutorials available?",
            "Is there a community forum where I can ask questions?",
            "Do you have a mobile app for iOS and Android?",
            "What programming languages does your SDK support?",
            "Where can I find your changelog or release notes?",
            "Do you offer webinars or training sessions?",
            "Can I try a demo before purchasing?",
        ],
    },
}

records = []
for category, priorities in ticket_templates.items():
    for priority, texts in priorities.items():
        for text in texts:
            # Add slight variation
            variations = [
                text,
                text + " Please help.",
                "Hi support team, " + text.lower(),
                text + " Thank you.",
                "Urgent: " + text if priority == "High" else text,
            ]
            for v in variations:
                records.append({
                    "ticket_id": f"TKT-{len(records)+1000}",
                    "text": v,
                    "category": category,
                    "priority": priority,
                })

# Duplicate to get a realistic size
df_base = pd.DataFrame(records)
df = pd.concat([df_base] * 4, ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["ticket_id"] = [f"TKT-{1000+i}" for i in range(len(df))]

print(f"✅ Dataset created: {len(df)} tickets")
print(f"\nCategory distribution:\n{df['category'].value_counts()}")
print(f"\nPriority distribution:\n{df['priority'].value_counts()}")


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

STOPWORDS = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it","its",
    "itself","they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","these","those","am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing","a","an","the","and","but",
    "if","or","because","as","until","while","of","at","by","for","with","about",
    "against","between","into","through","during","before","after","above","below",
    "to","from","up","down","in","out","on","off","over","under","again","further",
    "then","once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own","same",
    "so","than","too","very","s","t","can","will","just","don","should","now","d",
    "ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn","hi","hello","dear","please","thank","thanks","team","support",
    "help","urgent","issue","problem",
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)           # remove URLs
    text = re.sub(r'\$[\d,.]+', 'MONEY', text)           # normalize money
    text = re.sub(r'\d+', 'NUM', text)                   # normalize numbers
    text = re.sub(r'[^\w\s]', ' ', text)                 # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
print("\n✅ Text preprocessing complete")
print("\nSample cleaned tickets:")
for _, row in df.sample(3, random_state=1).iterrows():
    print(f"  Original : {row['text'][:80]}")
    print(f"  Cleaned  : {row['clean_text'][:80]}")
    print()


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION & MODEL TRAINING
# ─────────────────────────────────────────────

# --- Category Classification ---
X = df['clean_text']
y_cat = df['category']
y_pri = df['priority']

X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
    X, y_cat, y_pri, test_size=0.2, random_state=42, stratify=y_cat
)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    sublinear_tf=True,
)

# --- Train Category Models ---
print("=" * 55)
print("CATEGORY CLASSIFICATION")
print("=" * 55)

category_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
}

cat_results = {}
for name, clf in category_models.items():
    pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])
    pipe.fit(X_train, yc_train)
    preds = pipe.predict(X_test)
    acc  = accuracy_score(yc_test, preds)
    f1   = f1_score(yc_test, preds, average='weighted')
    cat_results[name] = {"pipeline": pipe, "preds": preds, "acc": acc, "f1": f1}
    print(f"  {name:<25}  Acc={acc:.4f}  F1={f1:.4f}")

best_cat_name = max(cat_results, key=lambda n: cat_results[n]['f1'])
best_cat_pipe = cat_results[best_cat_name]['pipeline']
print(f"\n🏆 Best category model: {best_cat_name}")

# --- Train Priority Models ---
print("\n" + "=" * 55)
print("PRIORITY CLASSIFICATION")
print("=" * 55)

priority_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
}

pri_results = {}
for name, clf in priority_models.items():
    pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])
    pipe.fit(X_train, yp_train)
    preds = pipe.predict(X_test)
    acc  = accuracy_score(yp_test, preds)
    f1   = f1_score(yp_test, preds, average='weighted')
    pri_results[name] = {"pipeline": pipe, "preds": preds, "acc": acc, "f1": f1}
    print(f"  {name:<25}  Acc={acc:.4f}  F1={f1:.4f}")

best_pri_name = max(pri_results, key=lambda n: pri_results[n]['f1'])
best_pri_pipe = pri_results[best_pri_name]['pipeline']
print(f"\n🏆 Best priority model: {best_pri_name}")


# ─────────────────────────────────────────────
# 4. DETAILED EVALUATION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 55)

best_cat_preds = cat_results[best_cat_name]['preds']
best_pri_preds = pri_results[best_pri_name]['preds']

print("\n📊 Category Classification Report:")
print(classification_report(yc_test, best_cat_preds))

print("\n📊 Priority Classification Report:")
print(classification_report(yp_test, best_pri_preds))


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS  (single figure, 6 panels)
# ─────────────────────────────────────────────

cats   = sorted(df['category'].unique())
pris   = ['High', 'Medium', 'Low']
models = list(cat_results.keys())

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

ACCENT  = '#00d4ff'
GREEN   = '#00ff88'
ORANGE  = '#ff8c00'
RED     = '#ff4444'
BG      = '#1a1d2e'
TEXT    = '#e8e8f0'

pri_colors  = {'High': RED, 'Medium': ORANGE, 'Low': GREEN}
cat_palette = [ACCENT, '#a78bfa', ORANGE, GREEN]

def style_ax(ax, title):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)

# — Panel 1: Category Distribution
ax1 = fig.add_subplot(gs[0, 0])
cat_counts = df['category'].value_counts()
bars = ax1.barh(cat_counts.index, cat_counts.values, color=cat_palette, edgecolor='none', height=0.6)
for bar, val in zip(bars, cat_counts.values):
    ax1.text(val + 5, bar.get_y() + bar.get_height()/2, str(val),
             va='center', color=TEXT, fontsize=9, fontweight='bold')
style_ax(ax1, "Ticket Volume by Category")
ax1.set_xlabel("Count")

# — Panel 2: Priority Distribution
ax2 = fig.add_subplot(gs[0, 1])
pri_counts = df['priority'].value_counts()
colors_pie = [pri_colors[p] for p in pri_counts.index]
wedges, texts, autotexts = ax2.pie(
    pri_counts.values, labels=pri_counts.index,
    colors=colors_pie, autopct='%1.1f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(linewidth=2, edgecolor='#0f1117'),
)
for t in texts + autotexts:
    t.set_color(TEXT); t.set_fontsize(9)
style_ax(ax2, "Priority Distribution")
ax2.set_facecolor('#0f1117')

# — Panel 3: Model Accuracy Comparison
ax3 = fig.add_subplot(gs[0, 2])
x  = np.arange(len(models))
w  = 0.35
cat_accs = [cat_results[m]['acc'] for m in models]
pri_accs = [pri_results[m]['acc'] for m in models]
b1 = ax3.bar(x - w/2, cat_accs, w, label='Category', color=ACCENT, alpha=0.85, edgecolor='none')
b2 = ax3.bar(x + w/2, pri_accs, w, label='Priority',  color='#a78bfa', alpha=0.85, edgecolor='none')
ax3.set_xticks(x)
ax3.set_xticklabels(['LR', 'RF', 'NB'], color=TEXT, fontsize=9)
ax3.set_ylim(0.5, 1.05)
ax3.axhline(0.9, color=GREEN, linestyle='--', alpha=0.4, linewidth=1)
for bar in list(b1) + list(b2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.2f}', ha='center', va='bottom', color=TEXT, fontsize=7.5)
ax3.legend(facecolor=BG, edgecolor='#333355', labelcolor=TEXT, fontsize=8)
style_ax(ax3, "Model Accuracy Comparison")
ax3.set_ylabel("Accuracy")

# — Panel 4: Category Confusion Matrix
ax4 = fig.add_subplot(gs[1, :2])
cm_cat = confusion_matrix(yc_test, best_cat_preds, labels=cats)
sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues',
            xticklabels=[c[:8] for c in cats],
            yticklabels=[c[:8] for c in cats],
            ax=ax4, linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax4.set_facecolor(BG)
ax4.tick_params(colors=TEXT, labelsize=9)
ax4.set_title(f"Category Confusion Matrix — {best_cat_name}", color=TEXT, fontsize=11, fontweight='bold', pad=10)
ax4.set_ylabel("True", color=TEXT); ax4.set_xlabel("Predicted", color=TEXT)
ax4.collections[0].colorbar.ax.tick_params(colors=TEXT)

# — Panel 5: Priority Confusion Matrix
ax5 = fig.add_subplot(gs[1, 2])
cm_pri = confusion_matrix(yp_test, best_pri_preds, labels=pris)
sns.heatmap(cm_pri, annot=True, fmt='d', cmap='Purples',
            xticklabels=pris, yticklabels=pris,
            ax=ax5, linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax5.set_facecolor(BG)
ax5.tick_params(colors=TEXT, labelsize=9)
ax5.set_title(f"Priority Confusion Matrix — {best_pri_name}", color=TEXT, fontsize=11, fontweight='bold', pad=10)
ax5.set_ylabel("True", color=TEXT); ax5.set_xlabel("Predicted", color=TEXT)
ax5.collections[0].colorbar.ax.tick_params(colors=TEXT)

# — Panel 6: Per-class F1 Scores
ax6 = fig.add_subplot(gs[2, 0])
from sklearn.metrics import f1_score as f1
cat_f1s = [f1(yc_test, best_cat_preds, labels=[c], average=None)[0] for c in cats]
bars6 = ax6.bar(range(len(cats)), cat_f1s, color=cat_palette, edgecolor='none')
ax6.set_xticks(range(len(cats)))
ax6.set_xticklabels([c.replace(' ', '\n') for c in cats], fontsize=8)
ax6.set_ylim(0, 1.1)
for b, v in zip(bars6, cat_f1s):
    ax6.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
             ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
style_ax(ax6, "Per-Class F1 — Category")
ax6.set_ylabel("F1 Score")

# — Panel 7: Per-class F1 Priority
ax7 = fig.add_subplot(gs[2, 1])
pri_f1s = [f1(yp_test, best_pri_preds, labels=[p], average=None)[0] for p in pris]
colors7 = [pri_colors[p] for p in pris]
bars7 = ax7.bar(pris, pri_f1s, color=colors7, edgecolor='none')
ax7.set_ylim(0, 1.1)
for b, v in zip(bars7, pri_f1s):
    ax7.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
             ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
style_ax(ax7, "Per-Class F1 — Priority")
ax7.set_ylabel("F1 Score")

# — Panel 8: Category vs Priority heatmap
ax8 = fig.add_subplot(gs[2, 2])
cross = pd.crosstab(df['category'], df['priority'])[pris]
sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd',
            ax=ax8, linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax8.set_facecolor(BG)
ax8.tick_params(colors=TEXT, labelsize=8)
ax8.set_title("Category × Priority Heatmap", color=TEXT, fontsize=11, fontweight='bold', pad=10)
ax8.set_ylabel("Category", color=TEXT); ax8.set_xlabel("Priority", color=TEXT)
ax8.collections[0].colorbar.ax.tick_params(colors=TEXT)

fig.suptitle("Support Ticket Classification & Prioritization — ML Dashboard",
             color=TEXT, fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/home/claude/ml_dashboard.png', dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("\n✅ Dashboard saved → ml_dashboard.png")


# ─────────────────────────────────────────────
# 6. REAL-TIME PREDICTION DEMO
# ─────────────────────────────────────────────

def predict_ticket(text: str):
    cleaned = clean_text(text)
    category = best_cat_pipe.predict([cleaned])[0]
    priority = best_pri_pipe.predict([cleaned])[0]
    cat_proba = best_cat_pipe.predict_proba([cleaned])[0]
    pri_proba = best_pri_pipe.predict_proba([cleaned])[0]
    cat_conf  = max(cat_proba)
    pri_conf  = max(pri_proba)
    return {
        "text": text,
        "predicted_category": category,
        "category_confidence": round(cat_conf * 100, 1),
        "predicted_priority": priority,
        "priority_confidence": round(pri_conf * 100, 1),
        "action": _get_action(category, priority),
    }

def _get_action(category, priority):
    actions = {
        ("Technical Issue", "High"):  "🚨 Escalate to on-call engineer immediately",
        ("Billing",         "High"):  "💳 Route to billing team — process refund",
        ("Account",         "High"):  "🔐 Security team alert — account compromise",
        ("General Query",   "High"):  "📋 Account manager follow-up within 1 hour",
        ("Technical Issue", "Medium"):"⚙️  Assign to technical support queue (SLA: 4h)",
        ("Billing",         "Medium"):"💬 Billing specialist — respond within 4 hours",
        ("Account",         "Medium"):"👤 Account support — respond within 4 hours",
        ("General Query",   "Medium"):"📧 Standard support queue (SLA: 8h)",
        ("Technical Issue", "Low"):   "📝 Log in backlog — resolve within 3 days",
        ("Billing",         "Low"):   "📝 Self-service FAQ or respond within 3 days",
        ("Account",         "Low"):   "📝 Documentation link + 3-day response",
        ("General Query",   "Low"):   "📝 Auto-reply with FAQ link",
    }
    return actions.get((category, priority), "📬 Route to general support queue")

# Demo tickets
demo_tickets = [
    "The entire API is down and our production app is completely broken. All 10,000 users are affected.",
    "I was charged $299 instead of $29. Please fix this billing error.",
    "How do I update my profile picture in the app?",
    "Someone hacked my account and changed my password. I cannot log in.",
    "Do you offer annual billing discounts for teams?",
    "The mobile app crashes every time I open it after the latest update.",
]

print("\n" + "=" * 65)
print("REAL-TIME TICKET PREDICTION DEMO")
print("=" * 65)
for ticket in demo_tickets:
    result = predict_ticket(ticket)
    print(f"\n📨 Ticket  : {result['text'][:70]}")
    print(f"   Category : {result['predicted_category']} ({result['category_confidence']}%)")
    print(f"   Priority : {result['predicted_priority']} ({result['priority_confidence']}%)")
    print(f"   Action   : {result['action']}")

# ─────────────────────────────────────────────
# 7. SUMMARY STATS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 55)
print(f"  Dataset size     : {len(df):,} tickets")
print(f"  Training set     : {len(X_train):,} tickets")
print(f"  Test set         : {len(X_test):,}  tickets")
print(f"  Best Cat Model   : {best_cat_name}")
print(f"  Category Acc     : {cat_results[best_cat_name]['acc']:.4f}")
print(f"  Category F1 (W)  : {cat_results[best_cat_name]['f1']:.4f}")
print(f"  Best Pri Model   : {best_pri_name}")
print(f"  Priority Acc     : {pri_results[best_pri_name]['acc']:.4f}")
print(f"  Priority F1 (W)  : {pri_results[best_pri_name]['f1']:.4f}")
print("=" * 55)
print("\n✅ All done! Check ml_dashboard.png for visualizations.")
