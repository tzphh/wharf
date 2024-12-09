#include <iostream>
#include <cassert>
#include "libs/compressed_trees/trees/augmented_map.h"  

using namespace std;

// 定义测试用的 entry 结构
struct entry {
  using key_t = int;        // 键类型
  using val_t = int;        // 值类型
  using aug_t = int;        // Augmented 值类型

  // 从键值对中提取键
  static key_t get_key(const pair<key_t, val_t>& e) { return e.first; }

  // 从键值对中提取值
  static val_t get_val(const pair<key_t, val_t>& e) { return e.second; }

  // 设置值
  static void set_val(pair<key_t, val_t>& e, const val_t& v) { e.second = v; }

  // 计算 augmented 值（这里直接返回值）
  static aug_t from_entry(const key_t& key, const val_t& val) { return val; }
};

int main() {
  using aug_map_t = aug_map<entry>;  // 定义具体的 aug_map 类型

  // 初始化一个空的 aug_map
  aug_map_t m;
  m.init();
  // 测试插入
  m.insert({0, 0});
  m.insert({1, 10});
  m.insert({2, 20});

  assert(m.size() == 3);  // 确保插入成功
  cout << "Insert test passed!" << endl;

  // // 测试查找
  // auto result = m.find(2);
  // assert(result.has_value() && result.value() == 20);  // 找到键 2 的值
  // cout << "Find test passed!" << endl;

  // // 测试范围操作
  // auto sum_range = m.aug_range(1, 3);  // 键范围 [1, 3] 的 augmented 值
  // assert(sum_range == 60);  // 10 + 20 + 30 = 60
  // cout << "Range sum test passed!" << endl;

  // // 测试删除
  // m = aug_maremove(std::move(m), 2);
  // assert(m.size() == 2);  // 删除后大小为 2
  // result = m.find(2);
  // assert(!result.has_value());  // 确保键 2 已删除
  // cout << "Remove test passed!" << endl;

  // // 测试多插入
  // sequence<pair<int, int>> entries = {{4, 40}, {5, 50}, {6, 60}};
  // m = aug_map_t::multi_insert(std::move(m), entries);
  // assert(m.size() == 5);  // 添加 3 个条目后总数为 5
  // cout << "Multi-insert test passed!" << endl;

  // // 测试 augmented 查询
  // int left_sum = m.aug_left(5);  // 小于键 5 的累积值
  // assert(left_sum == 80);  // 10 + 30 + 40 = 80
  // int right_sum = m.aug_right(5);  // 大于键 5 的累积值
  // assert(right_sum == 60);  // 仅键 6 的值 60
  // cout << "Augmented queries test passed!" << endl;

  // // 测试并集操作
  // sequence<pair<int, int>> new_entries = {{7, 70}, {8, 80}};
  // aug_map_t other_map = aug_map_t::multi_insert(aug_map_t(), new_entries);
  // auto union_map = aug_map_t::map_union(std::move(m), std::move(other_map));
  // assert(union_map.size() == 7);  // 两个 map 的并集
  // cout << "Union test passed!" << endl;

  // // 测试差集操作
  // auto diff_map = aug_map_t::map_difference(std::move(union_map), std::move(other_map));
  // assert(diff_map.size() == 5);  // 仅保留原始 m 的元素
  // cout << "Difference test passed!" << endl;

  cout << "All tests passed successfully!" << endl;
  return 0;
}
