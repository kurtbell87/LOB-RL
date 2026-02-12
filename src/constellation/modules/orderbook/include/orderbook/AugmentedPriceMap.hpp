#pragma once

#include <optional>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @brief A self-contained AVL tree that stores (Key, Value) pairs,
 *        augmenting each node with 'subtree_size' to enable
 *        order-statistic operations in O(log N).
 *
 * This code does not depend on any external modules. All usage is internal
 * to the orderbook. Tests or external code do not include this directly.
 */
namespace constellation::modules::orderbook {

template <typename K, typename V>
class AugmentedPriceMap {
private:
  struct Node;

  /**
   * @brief NodePool for chunk-based node allocation, single-writer usage.
   */
  class NodePool {
  public:
    NodePool() = default;
    ~NodePool() {
      for (auto* block : blocks_) {
        delete[] block;
      }
      blocks_.clear();
    }
    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;

    Node* Allocate(const K& key, const V& value) {
      if (!free_list_) {
        const std::size_t BLOCK_SIZE = 256;
        Node* block = new Node[BLOCK_SIZE];
        blocks_.push_back(block);
        for (std::size_t i = 0; i < BLOCK_SIZE - 1; ++i) {
          block[i].pool_next = &block[i + 1];
        }
        block[BLOCK_SIZE - 1].pool_next = nullptr;
        free_list_ = &block[0];
      }
      Node* n = free_list_;
      free_list_ = free_list_->pool_next;
      n->key = key;
      n->value = value;
      n->left = nullptr;
      n->right = nullptr;
      n->height = 1;
      n->subtree_size = 1;
      n->pool_next = nullptr;
      return n;
    }

    void Deallocate(Node* node) {
      if (!node) return;
      node->pool_next = free_list_;
      free_list_ = node;
    }

  private:
    std::vector<Node*> blocks_;
    Node* free_list_{nullptr};
  };

  struct Node {
    K key;
    V value;
    Node* left;
    Node* right;
    int   height;
    size_t subtree_size;

    Node* pool_next; // singly-linked list usage for NodePool

    Node() : left(nullptr), right(nullptr),
             height(1), subtree_size(1), pool_next(nullptr) {}
  };

public:
  AugmentedPriceMap() : root_(nullptr) {}

  // no copy
  AugmentedPriceMap(const AugmentedPriceMap&) = delete;
  AugmentedPriceMap& operator=(const AugmentedPriceMap&) = delete;

  // move
  AugmentedPriceMap(AugmentedPriceMap&& other) noexcept {
    root_ = other.root_;
    other.root_ = nullptr;
  }
  AugmentedPriceMap& operator=(AugmentedPriceMap&& other) noexcept {
    if (this != &other) {
      clear(root_);
      root_ = other.root_;
      other.root_ = nullptr;
    }
    return *this;
  }

  ~AugmentedPriceMap() {
    clear(root_);
  }

  /**
   * @brief Insert (key, value). Overwrites value if key exists. O(log N).
   */
  void insert(const K& key, const V& value) {
    root_ = insertRec(root_, key, value);
  }

  /**
   * @brief Erase by `key`. Returns true if erased, false if not found.
   */
  bool erase(const K& key) {
    size_t old_sz = size();
    root_ = eraseRec(root_, key);
    return (size() < old_sz);
  }

  /**
   * @brief find() -> optional value
   */
  std::optional<V> find(const K& key) const {
    Node* cur = root_;
    while (cur) {
      if (key < cur->key) {
        cur = cur->left;
      } else if (key > cur->key) {
        cur = cur->right;
      } else {
        return cur->value;
      }
    }
    return std::nullopt;
  }

  /**
   * @brief number of nodes
   */
  size_t size() const {
    return root_ ? root_->subtree_size : 0;
  }

  /**
   * @brief get nth element (0-based) in sorted order
   */
  std::optional<std::pair<K, V>> nth(size_t n) const {
    if (n >= size()) return std::nullopt;
    return nthRec(root_, n);
  }

private:
  Node* root_;
  NodePool pool_;

  static int heightOf(Node* n) { return n ? n->height : 0; }
  static size_t sizeOf(Node* n) { return n ? n->subtree_size : 0; }

  static void updateMetadata(Node* n) {
    if (!n) return;
    int hl = heightOf(n->left);
    int hr = heightOf(n->right);
    n->height = 1 + (hl > hr ? hl : hr);
    n->subtree_size = 1 + sizeOf(n->left) + sizeOf(n->right);
  }

  static int getBalance(Node* n) {
    return n ? (heightOf(n->left) - heightOf(n->right)) : 0;
  }

  Node* newNode(const K& key, const V& val) {
    return pool_.Allocate(key, val);
  }
  void freeNode(Node* node) {
    pool_.Deallocate(node);
  }

  Node* rotateRight(Node* y) {
    Node* x = y->left;
    Node* T2= x->right;
    x->right = y;
    y->left  = T2;
    updateMetadata(y);
    updateMetadata(x);
    return x;
  }
  Node* rotateLeft(Node* x) {
    Node* y = x->right;
    Node* T2= y->left;
    y->left = x;
    x->right= T2;
    updateMetadata(x);
    updateMetadata(y);
    return y;
  }

  Node* insertRec(Node* n, const K& key, const V& val) {
    if (!n) {
      return newNode(key, val);
    }
    if (key < n->key) {
      n->left = insertRec(n->left, key, val);
    } else if (key > n->key) {
      n->right= insertRec(n->right, key, val);
    } else {
      // update
      n->value = val;
      return n;
    }
    updateMetadata(n);

    int balance = getBalance(n);
    // standard AVL rebalances
    if (balance > 1 && key < n->left->key) {
      return rotateRight(n);
    }
    if (balance < -1 && key > n->right->key) {
      return rotateLeft(n);
    }
    if (balance > 1 && key > n->left->key) {
      n->left = rotateLeft(n->left);
      return rotateRight(n);
    }
    if (balance < -1 && key < n->right->key) {
      n->right= rotateRight(n->right);
      return rotateLeft(n);
    }
    return n;
  }

  Node* eraseRec(Node* n, const K& key) {
    if (!n) return nullptr;
    if (key < n->key) {
      n->left = eraseRec(n->left, key);
    } else if (key > n->key) {
      n->right= eraseRec(n->right, key);
    } else {
      // matched
      if (!n->left || !n->right) {
        Node* temp = n->left ? n->left : n->right;
        freeNode(n);
        return temp;
      } else {
        Node* succ = minValueNode(n->right);
        n->key   = succ->key;
        n->value = succ->value;
        n->right = eraseRec(n->right, succ->key);
      }
    }
    if (!n) return n;
    updateMetadata(n);

    int balance = getBalance(n);
    if (balance > 1 && getBalance(n->left) >= 0) {
      return rotateRight(n);
    }
    if (balance > 1 && getBalance(n->left) < 0) {
      n->left = rotateLeft(n->left);
      return rotateRight(n);
    }
    if (balance < -1 && getBalance(n->right) <= 0) {
      return rotateLeft(n);
    }
    if (balance < -1 && getBalance(n->right) > 0) {
      n->right = rotateRight(n->right);
      return rotateLeft(n);
    }
    return n;
  }

  static Node* minValueNode(Node* n) {
    Node* cur = n;
    while (cur && cur->left) {
      cur = cur->left;
    }
    return cur;
  }

  std::optional<std::pair<K, V>> nthRec(Node* n, size_t idx) const {
    if (!n) return std::nullopt;
    size_t left_sz = sizeOf(n->left);
    if (idx == left_sz) {
      return std::make_pair(n->key, n->value);
    } else if (idx < left_sz) {
      return nthRec(n->left, idx);
    } else {
      return nthRec(n->right, idx - left_sz - 1);
    }
  }

  void clear(Node* n) {
    if (!n) return;
    clear(n->left);
    clear(n->right);
    freeNode(n);
  }
};

} // end namespace constellation::modules::orderbook
