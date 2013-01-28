#include <iostream>
#include <vector>
#include <hash_map>

// taken from: http://www.cs.uml.edu/~jlu1/doc/codes/lruCache.html
// modify with: http://timday.bitbucket.org/lru.html

template<class K, class V>
struct LRUCacheEntry 
{
				K key;
				V value;
				LRUCacheEntry* prev;
				LRUCacheEntry* next;
};

template<class K, class V>
class LRUCache
{
private: 
				hash_map< K, LRUCacheEntry
