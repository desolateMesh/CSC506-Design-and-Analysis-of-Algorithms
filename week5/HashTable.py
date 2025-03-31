# Python Implementation for Personalized Content Recommendation using Hash Tables

class HashTable:
    def __init__(self, size=1000):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, content):
        idx = self._hash(key)
        for i, (stored_key, stored_content) in enumerate(self.table[idx]):
            if stored_key == key:
                self.table[idx][i] = (key, content)
                return
        self.table[idx].append((key, content))

    def retrieve(self, key):
        idx = self._hash(key)
        for stored_key, content in self.table[idx]:
            if stored_key == key:
                return content
        return None


class RecommendationSystem:
    def __init__(self):
        self.user_preferences = HashTable()

    def update_preferences(self, user_id, content):
        existing_preferences = self.user_preferences.retrieve(user_id) or []
        existing_preferences.append(content)
        self.user_preferences.insert(user_id, existing_preferences)

    def recommend_content(self, user_id):
        return self.user_preferences.retrieve(user_id) or []



if __name__ == '__main__':
    rec_sys = RecommendationSystem()
    rec_sys.update_preferences('user789', 'Downhill Mountain Biking')
    rec_sys.update_preferences('user789', 'Fly Fishing')
    rec_sys.update_preferences('user101', 'Crane Operation')
    rec_sys.update_preferences('user101', 'Skateboarding')

    # Retrieve recommendations
    print("Recommendations for user789:", rec_sys.recommend_content('user789'))
    print("Recommendations for user101:", rec_sys.recommend_content('user101'))

