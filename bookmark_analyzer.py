"""
Bookmark Analyzer Module
Extracts topics, creates clusters, and builds knowledge graphs from bookmarks
"""

import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import string
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Topic:
    """Represents a topic/cluster of bookmarks"""
    name: str
    keywords: List[str]
    bookmark_ids: List[str]
    score: float = 0.0
    
@dataclass
class BookmarkNode:
    """Node in the knowledge graph"""
    bookmark_id: str
    text: str
    author: str
    topics: List[str]
    keywords: List[str]
    connections: List[str]  # IDs of related bookmarks


class BookmarkAnalyzer:
    """Analyzes bookmarks to extract topics and create knowledge graphs"""
    
    def __init__(self):
        # Common words to ignore
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'how', 'when', 'where', 'why', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'this',
            'that', 'all', 'will', 'just', 'can', 'rt', 'amp', 'https'
        }
        
        # Tech-related keywords for better categorization
        self.tech_categories = {
            'ai_ml': ['ai', 'ml', 'llm', 'gpt', 'claude', 'anthropic', 'openai', 
                      'machine learning', 'artificial intelligence', 'model', 'agent',
                      'neural', 'deep learning', 'transformer', 'chatbot'],
            'programming': ['python', 'javascript', 'typescript', 'code', 'coding',
                           'programming', 'developer', 'github', 'git', 'api',
                           'framework', 'library', 'function', 'class', 'debug'],
            'tools': ['tool', 'app', 'software', 'plugin', 'extension', 'ide',
                     'editor', 'vscode', 'cursor', 'browser', 'automation'],
            'education': ['course', 'tutorial', 'learn', 'guide', 'tips', 'tricks',
                         'how to', 'lesson', 'teach', 'education', 'university'],
            'web_dev': ['react', 'vue', 'angular', 'html', 'css', 'frontend',
                       'backend', 'fullstack', 'website', 'web', 'server'],
            'data': ['data', 'database', 'sql', 'nosql', 'analytics', 'bigdata',
                    'pandas', 'numpy', 'visualization', 'dataset'],
            'devops': ['docker', 'kubernetes', 'aws', 'cloud', 'deploy', 'ci/cd',
                      'pipeline', 'infrastructure', 'server', 'hosting']
        }
        
        self.bookmarks = []
        self.topics = {}
        self.knowledge_graph = {}
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and remove URLs
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove punctuation except for technical terms
        text = re.sub(r'[^\w\s\-\+\#\.\/]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        keywords.extend([h.lower() for h in hashtags])
        
        # Extract technical terms (e.g., "claude-3", "gpt-4")
        tech_terms = re.findall(r'\b[\w]+-[\w]+\b', text)
        keywords.extend([t.lower() for t in tech_terms])
        
        # Count frequency and return top keywords
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(10)]
    
    def categorize_bookmark(self, text: str, keywords: List[str]) -> List[str]:
        """Categorize bookmark based on keywords and content"""
        categories = []
        text_lower = text.lower()
        
        for category, terms in self.tech_categories.items():
            # Check if any category terms appear in text or keywords
            if any(term in text_lower for term in terms):
                categories.append(category)
            elif any(any(term in kw for term in terms) for kw in keywords):
                categories.append(category)
        
        # If no categories found, mark as 'general'
        if not categories:
            categories.append('general')
            
        return categories
    
    def analyze_bookmarks(self, bookmarks: List[Dict]) -> Dict:
        """Analyze all bookmarks and create topic clusters"""
        self.bookmarks = bookmarks
        bookmark_nodes = []
        
        # First pass: Extract keywords and create nodes
        for bookmark in bookmarks:
            text = bookmark.get('text', '')
            author = bookmark.get('author_username', 'unknown')
            bookmark_id = bookmark.get('id', '')
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Categorize
            categories = self.categorize_bookmark(text, keywords)
            
            # Create node
            node = BookmarkNode(
                bookmark_id=bookmark_id,
                text=text,
                author=author,
                topics=categories,
                keywords=keywords,
                connections=[]
            )
            bookmark_nodes.append(node)
            self.knowledge_graph[bookmark_id] = node
        
        # Second pass: Find connections between bookmarks
        for i, node1 in enumerate(bookmark_nodes):
            for j, node2 in enumerate(bookmark_nodes[i+1:], i+1):
                # Check for common keywords
                common_keywords = set(node1.keywords) & set(node2.keywords)
                if len(common_keywords) >= 2:  # At least 2 common keywords
                    node1.connections.append(node2.bookmark_id)
                    node2.connections.append(node1.bookmark_id)
                
                # Check for same author
                if node1.author == node2.author and node1.author != 'unknown':
                    if node2.bookmark_id not in node1.connections:
                        node1.connections.append(node2.bookmark_id)
                        node2.connections.append(node1.bookmark_id)
        
        # Create topic clusters
        self.create_topic_clusters(bookmark_nodes)
        
        return {
            'total_bookmarks': len(bookmarks),
            'topics': self.topics,
            'categories': self.get_category_stats(),
            'top_authors': self.get_top_authors(),
            'keyword_frequency': self.get_keyword_frequency()
        }
    
    def create_topic_clusters(self, nodes: List[BookmarkNode]):
        """Create topic clusters from bookmark nodes"""
        # Group by primary category
        category_groups = defaultdict(list)
        for node in nodes:
            if node.topics:
                primary_topic = node.topics[0]
                category_groups[primary_topic].append(node)
        
        # Create topic objects
        for category, nodes_in_category in category_groups.items():
            # Collect all keywords from bookmarks in this category
            all_keywords = []
            bookmark_ids = []
            
            for node in nodes_in_category:
                all_keywords.extend(node.keywords)
                bookmark_ids.append(node.bookmark_id)
            
            # Get most common keywords for this topic
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(20)]
            
            topic = Topic(
                name=category,
                keywords=top_keywords,
                bookmark_ids=bookmark_ids,
                score=len(bookmark_ids)  # Simple scoring based on number of bookmarks
            )
            
            self.topics[category] = topic
    
    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about bookmark categories"""
        stats = defaultdict(int)
        for node in self.knowledge_graph.values():
            for topic in node.topics:
                stats[topic] += 1
        return dict(stats)
    
    def get_top_authors(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most bookmarked authors"""
        author_counts = Counter()
        for node in self.knowledge_graph.values():
            if node.author != 'unknown':
                author_counts[node.author] += 1
        return author_counts.most_common(limit)
    
    def get_keyword_frequency(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get most common keywords across all bookmarks"""
        all_keywords = []
        for node in self.knowledge_graph.values():
            all_keywords.extend(node.keywords)
        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(limit)
    
    def get_bookmarks_by_topic(self, topic_name: str) -> List[Dict]:
        """Get all bookmarks for a specific topic"""
        if topic_name not in self.topics:
            return []
        
        topic = self.topics[topic_name]
        bookmarks = []
        
        for bookmark_id in topic.bookmark_ids:
            # Find original bookmark
            for b in self.bookmarks:
                if b.get('id') == bookmark_id:
                    bookmarks.append(b)
                    break
        
        return bookmarks
    
    def get_related_bookmarks(self, bookmark_id: str, limit: int = 5) -> List[Dict]:
        """Get bookmarks related to a specific bookmark"""
        if bookmark_id not in self.knowledge_graph:
            return []
        
        node = self.knowledge_graph[bookmark_id]
        related = []
        
        for related_id in node.connections[:limit]:
            for b in self.bookmarks:
                if b.get('id') == related_id:
                    related.append(b)
                    break
        
        return related
    
    def export_knowledge_graph_mermaid(self) -> str:
        """Export knowledge graph as Mermaid diagram"""
        mermaid = ["graph TD"]
        
        # Add topic nodes
        for topic_name, topic in self.topics.items():
            safe_name = topic_name.replace(' ', '_')
            mermaid.append(f"    {safe_name}[{topic_name} - {len(topic.bookmark_ids)} bookmarks]")
        
        # Add connections between topics based on shared bookmarks
        processed_pairs = set()
        for topic1_name, topic1 in self.topics.items():
            for topic2_name, topic2 in self.topics.items():
                if topic1_name != topic2_name:
                    pair = tuple(sorted([topic1_name, topic2_name]))
                    if pair not in processed_pairs:
                        # Check for shared keywords
                        shared = set(topic1.keywords) & set(topic2.keywords)
                        if len(shared) >= 3:
                            safe_name1 = topic1_name.replace(' ', '_')
                            safe_name2 = topic2_name.replace(' ', '_')
                            mermaid.append(f"    {safe_name1} --> {safe_name2}")
                            processed_pairs.add(pair)
        
        return '\n'.join(mermaid)
    
    def get_topic_summary(self) -> str:
        """Get a text summary of topics"""
        summary = []
        summary.append("# Bookmark Topics Summary\n")
        
        # Sort topics by number of bookmarks
        sorted_topics = sorted(self.topics.items(), 
                              key=lambda x: x[1].score, 
                              reverse=True)
        
        for topic_name, topic in sorted_topics:
            summary.append(f"\n## {topic_name.title()} ({len(topic.bookmark_ids)} bookmarks)")
            summary.append(f"**Keywords**: {', '.join(topic.keywords[:10])}")
            summary.append("")
        
        return '\n'.join(summary)