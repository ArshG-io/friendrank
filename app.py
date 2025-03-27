import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Friend Power Rankings",
    page_icon="👥",
    layout="wide",
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #1E88E5;
        font-size: 42px !important;
    }
    .subtitle {
        font-size: 26px !important;
        color: #424242;
    }
    .highlight {
        background-color: #ffff99;
        border-radius: 5px;
        padding: 0.2em 0.4em;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        background-color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# File paths for data storage - Modified for Streamlit Cloud
data_dir = "data"
friends_file = os.path.join(data_dir, "friends_data.json")
rankings_file = os.path.join(data_dir, "rankings_history.json")
self_evals_file = os.path.join(data_dir, "self_evaluations.json")

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load saved data or initialize
def load_data():
    # Load friends data
    if os.path.exists(friends_file):
        try:
            with open(friends_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading friends data: {e}")
            return []
    return []

def load_rankings_history():
    # Load rankings history
    if os.path.exists(rankings_file):
        try:
            with open(rankings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading rankings history: {e}")
            return {}
    return {}

# Load self-evaluations
def load_self_evals():
    if os.path.exists(self_evals_file):
        try:
            with open(self_evals_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading self-evaluations: {e}")
            return {}
    return {}

# Save data function
def save_friends_data():
    try:
        with open(friends_file, 'w') as f:
            json.dump(st.session_state.friends, f)
    except Exception as e:
        st.error(f"Error saving friends data: {e}")

def save_rankings_history():
    try:
        with open(rankings_file, 'w') as f:
            json.dump(st.session_state.rankings_history, f)
    except Exception as e:
        st.error(f"Error saving rankings history: {e}")
        
def save_self_evals():
    try:
        with open(self_evals_file, 'w') as f:
            json.dump(st.session_state.self_evals, f)
    except Exception as e:
        st.error(f"Error saving self-evaluations: {e}")

# Initialize session state for storing friends data
if 'friends' not in st.session_state:
    st.session_state.friends = load_data()

if 'rankings_history' not in st.session_state:
    st.session_state.rankings_history = load_rankings_history()
    
if 'self_evals' not in st.session_state:
    st.session_state.self_evals = load_self_evals()

# Title and description
st.markdown("<h1 class='title'>Friend Power Rankings</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Track, rank, and analyze your friendships!</h3>", unsafe_allow_html=True)

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Home", "Add Friends", "Rankings", "Visualizations", "Stats & Insights", "History", "How Rankings Work", "Self Evaluation", "Admin Panel"])

# Home page
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Welcome to Your Friend Power Rankings!")
    st.markdown("""
    This app helps you track and rank your friendships based on various factors.
    
    **Features:**
    * Add friends and their attributes
    * Rank your friends by different criteria
    * Visualize friendship rankings with interactive charts
    * Track changes in your friendships over time
    * Get insights about your social circle
    
    Get started by visiting the **Add Friends** section in the sidebar!
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.session_state.friends:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Your Top 3 Friends Right Now")
        
        # Sort friends by overall score
        sorted_friends = sorted(st.session_state.friends, key=lambda x: x.get('overall_score', 0) + x.get('vibe_adjustment', 0), reverse=True)[:3]
        
        cols = st.columns(3)
        for i, friend in enumerate(sorted_friends):
            with cols[i]:
                st.markdown(f"### {i+1}. {friend['name']}")
                adjusted_score = friend.get('overall_score', 0)
                if 'vibe_adjustment' in friend:
                    adjusted_score += friend['vibe_adjustment']
                st.markdown(f"**Score:** {adjusted_score}/100")
                st.markdown(f"**Best quality:** {friend.get('best_quality', 'N/A')}")
                st.image(f"https://ui-avatars.com/api/?name={friend['name'].replace(' ', '+')}&size=128&background=random", width=100)
        st.markdown("</div>", unsafe_allow_html=True)

# Add Friends page
elif page == "Add Friends":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Add a New Friend")
    
    with st.form("new_friend_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Friend's Name")
            friendship_length = st.number_input("Years of Friendship", min_value=0.0, max_value=100.0, step=0.5)
            frequency_of_contact = st.slider("Frequency of Contact", 1, 10, 5, help="1 = Rarely, 10 = Daily")
            reliability = st.slider("Reliability", 1, 10, 5, help="How dependable are they?")
            time_since_dart = st.selectbox("Time Since Last Dart", 
                                         ["Within Last Month", "Within Last 4 Months", 
                                          "Within Last 6 Months", "Within Last Year", 
                                          "Over a Year", "Never"],
                                         help="How long since you last smoked darts together")
            recent_vibes = st.text_area("Recent Slights/Vibes", help="Note any recent interactions, positive or negative")
        
        with col2:
            fun_factor = st.slider("Fun Factor", 1, 10, 5, help="How fun are they to be around?")
            emotional_support = st.slider("Emotional Support", 1, 10, 5, help="How supportive are they?")
            shared_interests = st.slider("Shared Interests", 1, 10, 5, help="How many interests do you share?")
            best_quality = st.text_input("Their Best Quality")
        
        notes = st.text_area("Notes (Optional)", height=100)
        
        submitted = st.form_submit_button("Add Friend")
        
        if submitted and name:
            # Calculate dart bonus based on recency
            dart_bonus = 0
            if time_since_dart == "Within Last Month":
                dart_bonus = 5
            elif time_since_dart == "Within Last 4 Months":
                dart_bonus = 4
            elif time_since_dart == "Within Last 6 Months":
                dart_bonus = 3
            elif time_since_dart == "Within Last Year":
                dart_bonus = 1
            
            # Calculate overall score - custom formula with dart bonus and vibe adjustment
            overall_score = round((reliability * 1.5 + emotional_support * 1.5 + 
                               fun_factor * 1.2 + frequency_of_contact + 
                               shared_interests + min(friendship_length * 0.5, 10) + 
                               dart_bonus) / 7.2 * 10)
                               
            # Trend indicator (neutral by default)
            trend = "neutral"
            
            new_friend = {
                'name': name,
                'friendship_length': friendship_length,
                'frequency_of_contact': frequency_of_contact,
                'reliability': reliability,
                'fun_factor': fun_factor,
                'emotional_support': emotional_support,
                'shared_interests': shared_interests,
                'best_quality': best_quality,
                'notes': notes,
                'recent_vibes': recent_vibes,
                'time_since_dart': time_since_dart,
                'overall_score': overall_score,
                'trend': trend,
                'added_date': datetime.now().strftime('%Y-%m-%d'),
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'vibe_adjustment': 0,  # Initialize vibe adjustment to 0
                'self_eval_submitted': False  # Track if they've submitted a self-evaluation
            }
            
            # Check if friend already exists
            friend_exists = False
            for i, friend in enumerate(st.session_state.friends):
                if friend['name'].lower() == name.lower():
                    st.session_state.friends[i] = new_friend
                    friend_exists = True
                    save_friends_data()
                    st.success(f"Updated {name}'s information!")
                    break
            
            if not friend_exists:
                st.session_state.friends.append(new_friend)
                save_friends_data()
                st.success(f"Added {name} to your friends list!")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display current friends
    if st.session_state.friends:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Your Friends")
        
        for i, friend in enumerate(st.session_state.friends):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(f"https://ui-avatars.com/api/?name={friend['name'].replace(' ', '+')}&size=128&background=random", width=80)
            
            with col2:
                st.markdown(f"### {friend['name']}")
                
                # Calculate overall score with vibe adjustment
                adjusted_score = friend.get('overall_score', 0)
                if 'vibe_adjustment' in friend:
                    adjusted_score += friend['vibe_adjustment']
                    
                st.markdown(f"**Overall Score:** {adjusted_score}/100")
                st.markdown(f"**Best Quality:** {friend['best_quality']}")
                
                # Show self-evaluation icon if they've submitted one
                if friend.get('self_eval_submitted', False):
                    st.markdown("📝 **Self-evaluation submitted**")
                
                # Display dart info if available
                if 'time_since_dart' in friend:
                    dart_emoji = "🚬" if friend['time_since_dart'] in ["Within Last Month", "Within Last 4 Months"] else "⏳"
                    st.markdown(f"**Last Dart:** {dart_emoji} {friend['time_since_dart']}")
                
                # Display recent vibes if available
                if 'recent_vibes' in friend and friend['recent_vibes']:
                    st.markdown(f"**Recent Vibes:** {friend['recent_vibes']}")
                
                if st.button(f"Remove {friend['name']}", key=f"remove_{i}"):
                    # Add to history before removing
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    if current_date not in st.session_state.rankings_history:
                        st.session_state.rankings_history[current_date] = []
                    
                    st.session_state.rankings_history[current_date].append({
                        'name': friend['name'],
                        'score': friend['overall_score'],
                        'status': 'Removed'
                    })
                    
                    st.session_state.friends.pop(i)
                    save_friends_data()
                    st.experimental_rerun()
            
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("You haven't added any friends yet. Use the form above to add your first friend!")

# Rankings page
elif page == "Rankings":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Friend Rankings")
    
    if not st.session_state.friends:
        st.info("You need to add friends first to see rankings!")
    else:
        # Rank friends options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ranking_option = st.selectbox(
                "Rank by",
                ["Overall Score", "Reliability", "Emotional Support", "Fun Factor", 
                 "Frequency of Contact", "Shared Interests", "Friendship Length"]
            )
        
        with col2:
            # Option to update friend trends
            if st.button("Update Friend Trends"):
                # Get previous rankings if they exist
                dates = list(st.session_state.rankings_history.keys())
                dates.sort(reverse=True)
                
                if len(dates) >= 2:
                    latest_date = dates[0]
                    previous_date = dates[1]
                    
                    # Get latest and previous rankings for overall score
                    latest_rankings = {}
                    previous_rankings = {}
                    
                    for entry in st.session_state.rankings_history[latest_date]:
                        if entry.get('ranking_type') == 'Overall Score':
                            latest_rankings[entry.get('name')] = entry.get('rank')
                    
                    for entry in st.session_state.rankings_history[previous_date]:
                        if entry.get('ranking_type') == 'Overall Score':
                            previous_rankings[entry.get('name')] = entry.get('rank')
                    
                    # Update trends for each friend
                    for i, friend in enumerate(st.session_state.friends):
                        name = friend['name']
                        if name in latest_rankings and name in previous_rankings:
                            if latest_rankings[name] < previous_rankings[name]:
                                st.session_state.friends[i]['trend'] = "up"
                            elif latest_rankings[name] > previous_rankings[name]:
                                st.session_state.friends[i]['trend'] = "down"
                            else:
                                st.session_state.friends[i]['trend'] = "neutral"
                    
                    save_friends_data()
                    st.success("Updated friend trends!")
        
        # Map selection to the actual key in friends data
        ranking_key_map = {
            "Overall Score": "overall_score",
            "Reliability": "reliability",
            "Emotional Support": "emotional_support",
            "Fun Factor": "fun_factor",
            "Frequency of Contact": "frequency_of_contact",
            "Shared Interests": "shared_interests",
            "Friendship Length": "friendship_length"
        }
        
        ranking_key = ranking_key_map[ranking_option]
        
        # Sort friends by selected ranking key, applying vibe adjustment if overall score
        if ranking_key == "overall_score":
            sorted_friends = sorted(st.session_state.friends, 
                                   key=lambda x: x.get(ranking_key, 0) + x.get('vibe_adjustment', 0), 
                                   reverse=True)
        else:
            sorted_friends = sorted(st.session_state.friends, 
                                   key=lambda x: x.get(ranking_key, 0), 
                                   reverse=True)
        
        # Display rankings
        for rank, friend in enumerate(sorted_friends, 1):
            # Get score based on ranking key
            if ranking_key == "overall_score":
                score = friend.get(ranking_key, 0)
                if 'vibe_adjustment' in friend:
                    score += friend['vibe_adjustment']
            else:
                score = friend.get(ranking_key, 0)
            
            # Create a score bar
            if ranking_key in ["reliability", "emotional_support", "fun_factor", 
                              "frequency_of_contact", "shared_interests"]:
                max_score = 10
            elif ranking_key == "overall_score":
                max_score = 100
            else:  # friendship_length
                max_score = max([f.get("friendship_length", 0) for f in st.session_state.friends]) or 1
            
            # Normalize score for bar
            normalized_score = min(score / max_score, 1.0)
            
            # Columns for layout
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"### #{rank}")
                st.image(f"https://ui-avatars.com/api/?name={friend['name'].replace(' ', '+')}&size=128&background=random", width=80)
            
            with col2:
                st.markdown(f"### {friend['name']}")
                st.markdown(f"**{ranking_option}:** {score}" + (" years" if ranking_key == "friendship_length" else ""))
                
                # Progress bar for score
                st.progress(normalized_score)
                
                # Additional info
                st.markdown(f"**Best Quality:** {friend.get('best_quality', 'N/A')}")
                
                # Show recent vibes if available
                if friend.get('recent_vibes'):
                    st.markdown(f"**Recent Vibes:** {friend.get('recent_vibes')}")
                
                # Show dart info if available
                if friend.get('time_since_dart'):
                    dart_emoji = "🚬" if friend.get('time_since_dart') in ["Within Last Month", "Within Last 4 Months"] else "⏳"
                    st.markdown(f"**Last Dart:** {dart_emoji} {friend.get('time_since_dart')}")
                    
                if friend.get('notes'):
                    st.markdown(f"**Notes:** {friend.get('notes')}")
            
            st.markdown("---")
        
        # Save current rankings to history
        if st.button("Save Current Rankings to History"):
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            if current_date not in st.session_state.rankings_history:
                st.session_state.rankings_history[current_date] = []
            
            for rank, friend in enumerate(sorted_friends, 1):
                # Get score based on ranking key (apply vibe adjustment if overall score)
                if ranking_key == "overall_score":
                    score = friend.get(ranking_key, 0)
                    if 'vibe_adjustment' in friend:
                        score += friend['vibe_adjustment']
                else:
                    score = friend.get(ranking_key, 0)
                
                st.session_state.rankings_history[current_date].append({
                    'name': friend['name'],
                    'score': score,
                    'rank': rank,
                    'ranking_type': ranking_option,
                    'status': 'Active'
                })
            
            save_rankings_history()
            st.success("Rankings saved to history!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Visualizations page
elif page == "Visualizations":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Friendship Visualizations")
    
    if not st.session_state.friends:
        st.info("You need to add friends first to see visualizations!")
    else:
        viz_type = st.radio(
            "Select Visualization",
            ["Overall Ranking", "Comparison by Attributes", "Friendship Length vs. Score", "Radar Chart"]
        )
        
        if viz_type == "Overall Ranking":
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort friends by overall score (including vibe adjustment)
            sorted_friends = sorted(st.session_state.friends, 
                                   key=lambda x: x.get('overall_score', 0) + x.get('vibe_adjustment', 0), 
                                   reverse=False)
            
            names = [friend['name'] for friend in sorted_friends]
            scores = [friend.get('overall_score', 0) + friend.get('vibe_adjustment', 0) for friend in sorted_friends]
            
            # Generate colors based on score (higher score = darker green)
            colors = [(0.0, 0.5 + (score/200), 0.0) for score in scores]
            
            bars = ax.barh(names, scores, color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Overall Score')
            ax.set_title('Friend Power Rankings')
            
            # Add score labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                        ha='left', va='center', fontweight='bold')
            
            st.pyplot(fig)
        
        elif viz_type == "Comparison by Attributes":
            # Select friends to compare
            friend_names = [friend['name'] for friend in st.session_state.friends]
            selected_friends = st.multiselect("Select friends to compare", friend_names, 
                                             default=friend_names[:min(3, len(friend_names))])
            
            if selected_friends:
                # Get data for selected friends
                selected_data = []
                for name in selected_friends:
                    for friend in st.session_state.friends:
                        if friend['name'] == name:
                            selected_data.append(friend)
                
                # Create dataframe for plotting
                df = pd.DataFrame([
                    {
                        'Friend': friend['name'],
                        'Reliability': friend.get('reliability', 0),
                        'Emotional Support': friend.get('emotional_support', 0),
                        'Fun Factor': friend.get('fun_factor', 0),
                        'Frequency of Contact': friend.get('frequency_of_contact', 0),
                        'Shared Interests': friend.get('shared_interests', 0)
                    }
                    for friend in selected_data
                ])
                
                # Melt the dataframe for easier plotting
                df_melted = pd.melt(df, id_vars=['Friend'], var_name='Attribute', value_name='Score')
                
                # Create the grouped bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get unique attributes and friends
                attributes = df_melted['Attribute'].unique()
                friends = df_melted['Friend'].unique()
                
                # Set width of bars
                bar_width = 0.15
                
                # Set position of bars on x axis
                r = np.arange(len(attributes))
                
                # Make the plot
                for i, friend in enumerate(friends):
                    data = df_melted[df_melted['Friend'] == friend]
                    ax.bar(r + i*bar_width, data['Score'], width=bar_width, label=friend)
                
                # Add labels and title
                ax.set_xticks(r + bar_width * (len(friends) - 1) / 2)
                ax.set_xticklabels(attributes, rotation=45, ha='right')
                ax.set_ylim(0, 10)
                ax.set_ylabel('Score')
                ax.set_title('Friend Comparison by Attributes')
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least one friend to compare.")
        
        elif viz_type == "Friendship Length vs. Score":
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data
            names = [friend['name'] for friend in st.session_state.friends]
            lengths = [friend.get('friendship_length', 0) for friend in st.session_state.friends]
            scores = [friend.get('overall_score', 0) + friend.get('vibe_adjustment', 0) for friend in st.session_state.friends]
            
            # Create scatter plot
            scatter = ax.scatter(lengths, scores, s=100, alpha=0.7, c=scores, cmap='viridis')
            
            # Add labels for each point
            for i, name in enumerate(names):
                ax.annotate(name, (lengths[i], scores[i]), 
                            xytext=(5, 5), textcoords='offset points')
            
            # Add trend line
            if len(lengths) > 1:
                z = np.polyfit(lengths, scores, 1)
                p = np.poly1d(z)
                ax.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8)
                
                # Calculate correlation
                corr = np.corrcoef(lengths, scores)[0, 1]
                ax.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')
            
            # Labels and title
            ax.set_xlabel('Friendship Length (Years)')
            ax.set_ylabel('Overall Score')
            ax.set_title('Relationship Between Friendship Length and Score')
            
            # Add colorbar
            plt.colorbar(scatter, label='Score')
            
            st.pyplot(fig)
        
        elif viz_type == "Radar Chart":
            # Select friends for radar chart
            friend_names = [friend['name'] for friend in st.session_state.friends]
            selected_friend = st.selectbox("Select a friend", friend_names)
            
            # Get friend data
            friend_data = None
            for friend in st.session_state.friends:
                if friend['name'] == selected_friend:
                    friend_data = friend
                    break
            
            if friend_data:
                # Categories for radar chart
                categories = ['Reliability', 'Emotional Support', 'Fun Factor', 
                             'Frequency of Contact', 'Shared Interests']
                
                # Get values
                values = [
                    friend_data.get('reliability', 0),
                    friend_data.get('emotional_support', 0),
                    friend_data.get('fun_factor', 0),
                    friend_data.get('frequency_of_contact', 0),
                    friend_data.get('shared_interests', 0)
                ]
                
                # Close the polygon
                values.append(values[0])
                categories.append(categories[0])
                
                # Create radar chart
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, polar=True)
                
                # Compute angle for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop
                
                # Plot data
                ax.plot(angles, values, 'o-', linewidth=2, label=selected_friend)
                ax.fill(angles, values, alpha=0.25)
                
                # Set category labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories[:-1])
                
                # Set radial limits
                ax.set_ylim(0, 10)
                
                # Add title
                plt.title(f"{selected_friend}'s Friendship Profile", size=15)
                
                st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Stats & Insights page
elif page == "Stats & Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Friendship Stats & Insights")
    
    if not st.session_state.friends:
        st.info("You need to add friends first to see stats and insights!")
    else:
        # Calculate some stats
        num_friends = len(st.session_state.friends)
        avg_score = sum((friend.get('overall_score', 0) + friend.get('vibe_adjustment', 0)) for friend in st.session_state.friends) / num_friends if num_friends else 0
        avg_friendship_length = sum(friend.get('friendship_length', 0) for friend in st.session_state.friends) / num_friends if num_friends else 0
        
        # Display summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Friends", num_friends)
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        with col3:
            st.metric("Average Friendship Length", f"{avg_friendship_length:.1f} years")
        
        # Calculate attribute averages
        attr_averages = {
            'Reliability': sum(friend.get('reliability', 0) for friend in st.session_state.friends) / num_friends,
            'Emotional Support': sum(friend.get('emotional_support', 0) for friend in st.session_state.friends) / num_friends,
            'Fun Factor': sum(friend.get('fun_factor', 0) for friend in st.session_state.friends) / num_friends,
            'Frequency of Contact': sum(friend.get('frequency_of_contact', 0) for friend in st.session_state.friends) / num_friends,
            'Shared Interests': sum(friend.get('shared_interests', 0) for friend in st.session_state.friends) / num_friends
        }
        
        # Calculate dart stats
        dart_stats = {
            "Within Last Month": 0,
            "Within Last 4 Months": 0,
            "Within Last 6 Months": 0,
            "Within Last Year": 0,
            "Over a Year": 0,
            "Never": 0
        }
        
        for friend in st.session_state.friends:
            if 'time_since_dart' in friend and friend['time_since_dart'] in dart_stats:
                dart_stats[friend['time_since_dart']] += 1
                
        # Plot attribute averages
        fig, ax = plt.subplots(figsize=(10, 5))
        attributes = list(attr_averages.keys())
        values = list(attr_averages.values())
        
        bars = ax.bar(attributes, values, color='skyblue')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Attributes')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores Across Friendship Attributes')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display dart stats
        st.markdown("### Friend Distribution by Last Dart Smoked Together")
        
        # Filter out zero values
        dart_data = {k: v for k, v in dart_stats.items() if v > 0}
        
        if dart_data:
            fig, ax = plt.subplots(figsize=(10, 5))
            categories = list(dart_data.keys())
            values = list(dart_data.values())
            
            # Create a color gradient (more recent = darker)
            all_categories = ["Within Last Month", "Within Last 4 Months", "Within Last 6 Months", 
                             "Within Last Year", "Over a Year", "Never"]
            colors = []
            
            for cat in categories:
                # Get position in the timeline (0 = most recent, 5 = never)
                pos = all_categories.index(cat)
                # Create color (darker = more recent)
                colors.append((0.2, 0.4, 0.6, 1 - (pos / 10)))
                
            bars = ax.bar(categories, values, color=colors)
            ax.set_xlabel('Time Since Last Dart')
            ax.set_ylabel('Number of Friends')
            ax.set_title('Distribution of Friends by Last Dart Smoked')
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No dart smoking data available yet.")
        
        # Friend distribution by score ranges
        st.markdown("### Friend Distribution by Score Ranges")
        
        score_ranges = {
            '90-100': 0,
            '80-89': 0,
            '70-79': 0,
            '60-69': 0,
            '50-59': 0,
            'Under 50': 0
        }
        
        for friend in st.session_state.friends:
            score = friend.get('overall_score', 0)
            if 'vibe_adjustment' in friend:
                score += friend['vibe_adjustment']
                
            if score >= 90:
                score_ranges['90-100'] += 1
            elif score >= 80:
                score_ranges['80-89'] += 1
            elif score >= 70:
                score_ranges['70-79'] += 1
            elif score >= 60:
                score_ranges['60-69'] += 1
            elif score >= 50:
                score_ranges['50-59'] += 1
            else:
                score_ranges['Under 50'] += 1
        
        # Create distribution chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ranges = list(score_ranges.keys())
        counts = list(score_ranges.values())
        
        # Add colors based on ranges
        colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#F44336']
        
        bars = ax.bar(ranges, counts, color=colors)
        ax.set_xlabel('Score Range')
        ax.set_ylabel('Number of Friends')
        ax.set_title('Distribution of Friends by Score Range')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there are friends in that range
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Generate insights
        st.markdown("### Friendship Insights")
        
        # Find best and worst attributes
        best_attr = max(attr_averages.items(), key=lambda x: x[1])
        worst_attr = min(attr_averages.items(), key=lambda x: x[1])
        
        st.markdown(f"🌟 **Strongest friendship quality:** {best_attr[0]} (avg. {best_attr[1]:.1f}/10)")
        st.markdown(f"🔍 **Area for improvement:** {worst_attr[0]} (avg. {worst_attr[1]:.1f}/10)")
        
        # Find correlation between friendship length and score
        lengths = [friend.get('friendship_length', 0) for friend in st.session_state.friends]
        scores = [friend.get('overall_score', 0) + friend.get('vibe_adjustment', 0) for friend in st.session_state.friends]
        
        if len(lengths) > 1:
            corr = np.corrcoef(lengths, scores)[0, 1]
            if corr >= 0.7:
                st.markdown(f"📊 **Strong correlation found:** Longer friendships tend to have higher scores (correlation: {corr:.2f})")
            elif corr >= 0.3:
                st.markdown(f"📊 **Moderate correlation found:** Friendship length has some influence on scores (correlation: {corr:.2f})")
            elif corr <= -0.3:
                st.markdown(f"📊 **Negative correlation found:** Newer friendships actually tend to score higher (correlation: {corr:.2f})")
        
        # Find top qualities
        if num_friends >= 3:
            qualities = [friend.get('best_quality', '') for friend in st.session_state.friends if friend.get('best_quality')]
            if qualities:
                quality_counts = {}
                for q in qualities:
                    if q:
                        quality_counts[q] = quality_counts.get(q, 0) + 1
                
                most_common_quality = max(quality_counts.items(), key=lambda x: x[1]) if quality_counts else None
                if most_common_quality and most_common_quality[1] > 1:
                    st.markdown(f"🎭 **Common quality:** '{most_common_quality[0]}' appears in {most_common_quality[1]} of your friends")
    
    st.markdown("</div>", unsafe_allow_html=True)

# History page
elif page == "History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Rankings History")
    
    if not st.session_state.rankings_history:
        st.info("You haven't saved any rankings to history yet. Go to the Rankings page and click 'Save Current Rankings to History'.")
    else:
        # Display available dates
        dates = list(st.session_state.rankings_history.keys())
        dates.sort(reverse=True)  # Most recent first
        
        selected_date = st.selectbox("Select date to view", dates)
        
        if selected_date:
            st.markdown(f"### Rankings from {selected_date}")
            
            # Group by ranking type
            rankings_by_type = {}
            for entry in st.session_state.rankings_history[selected_date]:
                ranking_type = entry.get('ranking_type', 'Overall Score')
                if ranking_type not in rankings_by_type:
                    rankings_by_type[ranking_type] = []
                rankings_by_type[ranking_type].append(entry)
            
            # Display rankings by type
            if rankings_by_type:
                for ranking_type, entries in rankings_by_type.items():
                    st.markdown(f"#### {ranking_type} Rankings")
                    
                    # Sort by rank
                    sorted_entries = sorted(entries, key=lambda x: x.get('rank', 999))
                    
                    for entry in sorted_entries:
                        status = entry.get('status', 'Active')
                        
                        # Use different styling based on status
                        if status == 'Active':
                            st.markdown(f"**#{entry.get('rank', '-')}:** {entry.get('name')} - Score: {entry.get('score')}")
                        elif status == 'Removed':
                            st.markdown(f"~~**#{entry.get('rank', '-')}:** {entry.get('name')} - Score: {entry.get('score')}~~ (Removed)")
                    
                    st.markdown("---")
            else:
                st.warning(f"No ranking information available for {selected_date}")
            
            # Option to delete this history entry
            if st.button(f"Delete History from {selected_date}"):
                del st.session_state.rankings_history[selected_date]
                save_rankings_history()
                st.success(f"Deleted history from {selected_date}")
                st.experimental_rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
        
# How Rankings Work page
elif page == "How Rankings Work":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## How Friend Power Rankings Are Calculated")
    
    st.markdown("""
    ### Ranking Formula
    
    Your friends are ranked based on a comprehensive scoring system that takes into account multiple factors. The overall score (out of 100) is calculated using this weighted formula:
    
    ```
    Overall Score = ((Reliability * 1.5) + 
                     (Emotional Support * 1.5) + 
                     (Fun Factor * 1.2) + 
                     (Frequency of Contact) + 
                     (Shared Interests) + 
                     min(Friendship Length * 0.5, 10) +
                     Dart Bonus) / 7.2 * 10 + Vibe Adjustment
    ```
    
    ### Weighting Explanation
    
    * **Reliability** (1.5x weight): Being dependable is highly valued
    * **Emotional Support** (1.5x weight): Being there when needed matters most
    * **Fun Factor** (1.2x weight): Enjoying time together is important but not everything
    * **Frequency of Contact** (1x weight): Regular interaction matters
    * **Shared Interests** (1x weight): Common ground is important
    * **Friendship Length** (0.5x weight, capped at 10): Loyalty over time is valued but capped to avoid over-rewarding just for duration
    * **Dart Bonus** (0-5 points): Extra points based on how recently you smoked darts together
      * Within Last Month: 5 points
      * Within Last 4 Months: 4 points
      * Within Last 6 Months: 3 points
      * Within Last Year: 1 point
      * Over a Year/Never: 0 points
    * **Vibe Adjustment**: Manual adjustment based on overall vibes (only settable by admin)
    
    ### Recent Slights/Vibes
    
    The "Recent Slights/Vibes" field captures qualitative information about recent interactions. This information doesn't directly affect the numerical score but can provide context when reviewing rankings.
    
    ### Trend Indicators
    
    Trends show how a friend's ranking has changed compared to previous rankings:
    * 🔼 Upward trend: Friend has moved up in rankings
    * 🔽 Downward trend: Friend has moved down in rankings
    * — Neutral: No significant change
    
    ### Ranking Interpretation
    
    * **90-100**: Elite tier friendship
    * **80-89**: Excellent friend
    * **70-79**: Solid friendship
    * **60-69**: Good friend with some areas for improvement
    * **50-59**: Average friendship
    * **Below 50**: Relationship needs attention
    
    Remember that these rankings are just tools to help you reflect on your relationships. The most important aspects of friendship can't always be measured numerically!
    """)
    
    st.markdown("<div class='card' style='background-color: #e8f4f8; padding: 15px;'>", unsafe_allow_html=True)
    st.markdown("### Example Calculation")
    st.markdown("""
    Let's say you have a friend with these ratings:
    * Reliability: 8/10
    * Emotional Support: 7/10
    * Fun Factor: 9/10
    * Frequency of Contact: 6/10
    * Shared Interests: 8/10
    * Friendship Length: 5 years
    * Time Since Last Dart: Within Last 4 Months (4 points)
    
    Calculating their score:
    ```
    ((8 * 1.5) + (7 * 1.5) + (9 * 1.2) + 6 + 8 + min(5 * 0.5, 10) + 4) / 7.2 * 10
    = (12 + 10.5 + 10.8 + 6 + 8 + 2.5 + 4) / 7.2 * 10
    = 53.8 / 7.2 * 10
    = 7.47 * 10
    = 74.7
    ```
    
    Rounded to the nearest whole number: **75**
    
    This would place them in the "Solid friendship" category.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Self Evaluation Page
elif page == "Self Evaluation":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Friend Self-Evaluation Form")
    
    st.markdown("""
    This form allows friends to submit their own evaluation of your friendship. Share this page with your friends to see how they rate themselves!
    
    After submitting, their responses will be available for review in the Admin Panel.
    """)
    
    with st.form("self_evaluation_form"):
        # Friend identification
        friend_name = st.text_input("Your Name")
        friend_email = st.text_input("Your Email (for verification)")
        
        st.markdown("### How do you rate yourself as a friend?")
        
        # Self-ratings
        self_reliability = st.slider("Your Reliability", 1, 10, 8, help="How dependable do you think you are?")
        self_emotional_support = st.slider("Your Emotional Support", 1, 10, 8, help="How supportive do you think you are?")
        self_fun_factor = st.slider("Your Fun Factor", 1, 10, 8, help="How fun do you think you are to be around?")
        self_frequency = st.slider("Your Effort to Stay in Touch", 1, 10, 8, help="How much effort do you make to maintain the friendship?")
        self_dart_time = st.selectbox("When did we last smoke darts together?", 
                                     ["Within Last Month", "Within Last 4 Months", 
                                      "Within Last 6 Months", "Within Last Year", 
                                      "Over a Year", "Never"])
        
        # Additional info
        self_best_quality = st.text_input("What's your best quality as a friend?")
        self_shared_memory = st.text_area("Share your favorite memory of our friendship")
        self_improvement = st.text_area("How could you improve as a friend?")
        
        submitted = st.form_submit_button("Submit Self-Evaluation")
        
        if submitted and friend_name and friend_email:
            # Check if this friend exists in the system
            friend_exists = False
            friend_index = -1
            
            for i, friend in enumerate(st.session_state.friends):
                if friend['name'].lower() == friend_name.lower():
                    friend_exists = True
                    friend_index = i
                    break
            
            # Store self-evaluation regardless of whether the friend exists
            self_eval = {
                'name': friend_name,
                'email': friend_email,
                'reliability': self_reliability,
                'emotional_support': self_emotional_support,
                'fun_factor': self_fun_factor,
                'frequency': self_frequency,
                'dart_time': self_dart_time,
                'best_quality': self_best_quality,
                'shared_memory': self_shared_memory,
                'improvement': self_improvement,
                'submission_date': datetime.now().strftime('%Y-%m-%d'),
                'reviewed': False,
                'in_friends_list': friend_exists
            }
            
            # Initialize self_evals in session state if it doesn't exist
            if 'self_evals' not in st.session_state:
                st.session_state.self_evals = {}
            
            # Store evaluation under friend's name
            st.session_state.self_evals[friend_name] = self_eval
            
            # Mark in the friend's record that they've submitted a self-eval (if the friend exists)
            if friend_exists:
                st.session_state.friends[friend_index]['self_eval_submitted'] = True
                save_friends_data()
            
            # Save self-evaluation data
            save_self_evals()
            
            st.success("Thank you for submitting your self-evaluation! Your input has been recorded.")
    
    st.markdown("</div>", unsafe_allow_html=True)
# Admin Panel Page
elif page == "Admin Panel":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Admin Panel")
    
    # Password protection
    password = st.text_input("Enter admin password", type="password")
    
    if password == "ArshIsKing":
        st.success("Access granted! Welcome, admin.")
        
        # Load self-evaluations if they exist
        if 'self_evals' not in st.session_state:
            # Try to load from file
            if os.path.exists(self_evals_file):
                try:
                    with open(self_evals_file, 'r') as f:
                        st.session_state.self_evals = json.load(f)
                except Exception as e:
                    st.error(f"Error loading self-evaluations: {e}")
                    st.session_state.self_evals = {}
            else:
                st.session_state.self_evals = {}
        
        # Admin tabs
        admin_tab = st.radio("Admin Functions", ["Vibe Adjustments", "Self-Evaluations", "Friend Stats Override"])
        
        if admin_tab == "Vibe Adjustments":
            st.markdown("### Vibe Meter Adjustments")
            st.markdown("Adjust friend rankings based on overall vibes. This will directly modify their scores.")
            
            for i, friend in enumerate(st.session_state.friends):
                st.markdown(f"#### {friend['name']}")
                current_vibe = friend.get('vibe_adjustment', 0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_vibe = st.slider(
                        f"Vibe Adjustment for {friend['name']}", 
                        min_value=-20, 
                        max_value=20, 
                        value=current_vibe,
                        key=f"vibe_{i}",
                        help="Negative for bad vibes, positive for good vibes"
                    )
                
                with col2:
                    if st.button(f"Apply Vibe", key=f"apply_vibe_{i}"):
                        st.session_state.friends[i]['vibe_adjustment'] = new_vibe
                        save_friends_data()
                        st.success(f"Updated vibe adjustment for {friend['name']} to {new_vibe}")
                
                # Display current score info
                base_score = friend.get('overall_score', 0) - friend.get('vibe_adjustment', 0)
                adjusted_score = base_score + new_vibe
                st.markdown(f"Base Score: {base_score} → Adjusted Score: {adjusted_score}")
                st.markdown("---")
        
        elif admin_tab == "Self-Evaluations":
            st.markdown("### Friend Self-Evaluations")
            
            if not st.session_state.self_evals:
                st.info("No self-evaluations have been submitted yet.")
            else:
                # Show list of submitted evaluations
                evals = list(st.session_state.self_evals.values())
                
                # Sort by submission date (newest first)
                evals.sort(key=lambda x: x.get('submission_date', ''), reverse=True)
                
                # Filter options
                show_filter = st.radio("Show", ["All", "Unreviewed Only", "Reviewed Only"])
                
                filtered_evals = evals
                if show_filter == "Unreviewed Only":
                    filtered_evals = [e for e in evals if not e.get('reviewed', False)]
                elif show_filter == "Reviewed Only":
                    filtered_evals = [e for e in evals if e.get('reviewed', False)]
                
                for eval_data in filtered_evals:
                    st.markdown(f"#### {eval_data['name']} ({eval_data['submission_date']})")
                    
                    # Display evaluation data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Self-Assessment:**")
                        st.markdown(f"Reliability: {eval_data.get('reliability', 'N/A')}/10")
                        st.markdown(f"Emotional Support: {eval_data.get('emotional_support', 'N/A')}/10")
                        st.markdown(f"Fun Factor: {eval_data.get('fun_factor', 'N/A')}/10")
                        st.markdown(f"Contact Effort: {eval_data.get('frequency', 'N/A')}/10")
                        st.markdown(f"Last Dart: {eval_data.get('dart_time', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Additional Information:**")
                        st.markdown(f"Best Quality: {eval_data.get('best_quality', 'N/A')}")
                        st.markdown(f"Favorite Memory: {eval_data.get('shared_memory', 'N/A')}")
                        st.markdown(f"Self-Improvement: {eval_data.get('improvement', 'N/A')}")
                    
                    # Comparison with your assessment (if they exist in friends list)
                    found_friend = False
                    for friend in st.session_state.friends:
                        if friend['name'].lower() == eval_data['name'].lower():
                            found_friend = True
                            st.markdown("### Your Assessment vs. Their Self-Assessment")
                            
                            comparison_data = {
                                "Attribute": ["Reliability", "Emotional Support", "Fun Factor", "Contact Frequency"],
                                "Your Rating": [
                                    friend.get('reliability', 'N/A'),
                                    friend.get('emotional_support', 'N/A'),
                                    friend.get('fun_factor', 'N/A'),
                                    friend.get('frequency_of_contact', 'N/A')
                                ],
                                "Their Self-Rating": [
                                    eval_data.get('reliability', 'N/A'),
                                    eval_data.get('emotional_support', 'N/A'),
                                    eval_data.get('fun_factor', 'N/A'),
                                    eval_data.get('frequency', 'N/A')
                                ],
                                "Difference": [
                                    eval_data.get('reliability', 0) - friend.get('reliability', 0),
                                    eval_data.get('emotional_support', 0) - friend.get('emotional_support', 0),
                                    eval_data.get('fun_factor', 0) - friend.get('fun_factor', 0),
                                    eval_data.get('frequency', 0) - friend.get('frequency_of_contact', 0)
                                ]
                            }
                            
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Dart time comparison
                            st.markdown(f"**Dart Timing:**")
                            st.markdown(f"Your assessment: {friend.get('time_since_dart', 'N/A')}")
                            st.markdown(f"Their assessment: {eval_data.get('dart_time', 'N/A')}")
                            break
                    
                    if not found_friend:
                        st.info("This person is not in your friends list yet.")
                    
                    # Mark as reviewed button
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if not eval_data.get('reviewed', False):
                            if st.button("Mark as Reviewed", key=f"mark_{eval_data['name']}"):
                                st.session_state.self_evals[eval_data['name']]['reviewed'] = True
                                
                                # Save self-evaluations
                                save_self_evals()
                                
                                st.success(f"Marked {eval_data['name']}'s evaluation as reviewed.")
                                st.experimental_rerun()
                        else:
                            st.info("Reviewed")
                    
                    st.markdown("---")
        
        elif admin_tab == "Friend Stats Override":
            st.markdown("### Override Friend Statistics")
            st.markdown("Manually adjust ratings and attributes for each friend.")
            
            # Select friend to edit
            friend_names = [friend['name'] for friend in st.session_state.friends]
            selected_friend = st.selectbox("Select friend to edit", friend_names)
            
            # Find the selected friend
            selected_index = -1
            selected_data = None
            
            for i, friend in enumerate(st.session_state.friends):
                if friend['name'] == selected_friend:
                    selected_index = i
                    selected_data = friend
                    break
            
            if selected_data:
                st.markdown(f"### Editing {selected_friend}'s Stats")
                
                with st.form(f"edit_friend_{selected_index}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        reliability = st.slider("Reliability", 1, 10, int(selected_data.get('reliability', 5)))
                        emotional_support = st.slider("Emotional Support", 1, 10, int(selected_data.get('emotional_support', 5)))
                        fun_factor = st.slider("Fun Factor", 1, 10, int(selected_data.get('fun_factor', 5)))
                        frequency = st.slider("Frequency of Contact", 1, 10, int(selected_data.get('frequency_of_contact', 5)))
                    
                    with col2:
                        shared_interests = st.slider("Shared Interests", 1, 10, int(selected_data.get('shared_interests', 5)))
                        friendship_length = st.number_input("Years of Friendship", 
                                                         min_value=0.0, 
                                                         max_value=100.0, 
                                                         value=float(selected_data.get('friendship_length', 1.0)),
                                                         step=0.5)
                        time_since_dart = st.selectbox("Time Since Last Dart", 
                                                     ["Within Last Month", "Within Last 4 Months", 
                                                      "Within Last 6 Months", "Within Last Year", 
                                                      "Over a Year", "Never"],
                                                     index=["Within Last Month", "Within Last 4 Months", 
                                                            "Within Last 6 Months", "Within Last Year", 
                                                            "Over a Year", "Never"].index(selected_data.get('time_since_dart', "Never")))
                    
                    best_quality = st.text_input("Best Quality", selected_data.get('best_quality', ''))
                    recent_vibes = st.text_area("Recent Slights/Vibes", selected_data.get('recent_vibes', ''))
                    notes = st.text_area("Notes", selected_data.get('notes', ''))
                    
                    submitted = st.form_submit_button("Update Friend")
                    
                    if submitted:
                        # Calculate dart bonus
                        dart_bonus = 0
                        if time_since_dart == "Within Last Month":
                            dart_bonus = 5
                        elif time_since_dart == "Within Last 4 Months":
                            dart_bonus = 4
                        elif time_since_dart == "Within Last 6 Months":
                            dart_bonus = 3
                        elif time_since_dart == "Within Last Year":
                            dart_bonus = 1
                        
                        # Calculate new score with current vibe adjustment
                        vibe_adjustment = selected_data.get('vibe_adjustment', 0)
                        overall_score = round((reliability * 1.5 + emotional_support * 1.5 + 
                                           fun_factor * 1.2 + frequency + 
                                           shared_interests + min(friendship_length * 0.5, 10) + 
                                           dart_bonus) / 7.2 * 10)
                        
                        # Update friend data
                        st.session_state.friends[selected_index].update({
                            'reliability': reliability,
                            'emotional_support': emotional_support,
                            'fun_factor': fun_factor,
                            'frequency_of_contact': frequency,
                            'shared_interests': shared_interests,
                            'friendship_length': friendship_length,
                            'time_since_dart': time_since_dart,
                            'best_quality': best_quality,
                            'recent_vibes': recent_vibes,
                            'notes': notes,
                            'overall_score': overall_score,
                            'last_updated': datetime.now().strftime('%Y-%m-%d')
                        })
                        
                        save_friends_data()
                        st.success(f"Updated {selected_friend}'s information!")
    else:
        st.error("Incorrect password. Please try again.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add this to help with Streamlit Cloud deployment
if __name__ == "__main__":
    # The app is already running through Streamlit, so we don't need to call any additional functions
    pass
