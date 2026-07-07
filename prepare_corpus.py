"""
第二步：代码语料准备
本地生成高质量合成代码语料作为 CPT 数据，避免 HuggingFace 认证依赖。
目标：10-50MB 纯文本，涵盖 Python/Java/SpringBoot/文档
"""

import os
import json
import random
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = "data/corpus"
TARGET_SIZE_MB = 40  # 目标 ~40MB
SEED = 42
random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========== Python 代码模板 ==========
PYTHON_SAMPLES = [
    # 数据结构与算法
    lambda: f"""class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = ListNode(val)
    
    def reverse(self):
        prev = None
        cur = self.head
        while cur:
            next_node = cur.next
            cur.next = prev
            prev = cur
            cur = next_node
        self.head = prev
    
    def __str__(self):
        vals = []
        cur = self.head
        while cur:
            vals.append(str(cur.val))
            cur = cur.next
        return " -> ".join(vals) if vals else "None"
""",
    lambda: f"""def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
    lambda: f"""class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
""",
    lambda: f"""class Graph:
    def __init__(self):
        self.graph = {{}}
    
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def bfs(self, start):
        visited = set()
        queue = [start]
        visited.add(start)
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result
    
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        result = [start]
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        return result
""",
    # 多线程与并发
    lambda: f"""import threading
import queue
import time
import random

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.workers = []
        for _ in range(num_threads):
            worker = threading.Thread(target=self._worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        while True:
            func, args, kwargs = self.tasks.get()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Task error: {{e}}")
            finally:
                self.tasks.task_done()
    
    def submit(self, func, *args, **kwargs):
        self.tasks.put((func, args, kwargs))
    
    def wait_completion(self):
        self.tasks.join()

def demo():
    pool = ThreadPool(4)
    for i in range(8):
        pool.submit(lambda x: print(f"Task {{x}} done by {{threading.current_thread().name}}"), i)
    pool.wait_completion()
""",
    lambda: f"""import asyncio
import aiohttp
import json

async def fetch_url(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        return json.dumps({{"error": str(e)}})

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

async def main():
    urls = [
        "https://api.github.com",
        "https://httpbin.org/get",
        "https://jsonplaceholder.typicode.com/posts/1",
    ]
    results = await fetch_all(urls)
    for url, result in zip(urls, results):
        print(f"{{url}}: {{len(result)}} chars")

if __name__ == "__main__":
    asyncio.run(main())
""",
    # Web 框架 (Flask/FastAPI)
    lambda: f"""from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Task API", version="1.0.0")

class Task(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool = False

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None

tasks_db = []
counter = 1

@app.get("/tasks", response_model=List[Task])
async def list_tasks():
    return tasks_db

@app.post("/tasks", response_model=Task, status_code=201)
async def create_task(task: TaskCreate):
    global counter
    new_task = Task(id=counter, title=task.title, description=task.description)
    tasks_db.append(new_task)
    counter += 1
    return new_task

@app.get("/tasks/{{task_id}}", response_model=Task)
async def get_task(task_id: int):
    for task in tasks_db:
        if task.id == task_id:
            return task
    raise HTTPException(status_code=404, detail="Task not found")

@app.put("/tasks/{{task_id}}")
async def update_task(task_id: int, task: TaskCreate):
    for t in tasks_db:
        if t.id == task_id:
            t.title = task.title
            t.description = task.description
            return {{"message": "updated"}}
    raise HTTPException(status_code=404, detail="Task not found")

@app.delete("/tasks/{{task_id}}", status_code=204)
async def delete_task(task_id: int):
    global tasks_db
    tasks_db = [t for t in tasks_db if t.id != task_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
    # 数据处理
    lambda: f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(df):
    \"\"\"数据分析流水线\"\"\"
    print("=" * 50)
    print("数据集概览")
    print("=" * 50)
    print(f"Shape: {{df.shape}}")
    print(f"\\n列信息:")
    print(df.info())
    print(f"\\n基本统计:")
    print(df.describe())
    print(f"\\n缺失值:")
    print(df.isnull().sum())
    return df

def preprocess_data(df, target_col, test_size=0.2):
    \"\"\"数据预处理\"\"\"
    # 处理缺失值
    df = df.fillna(df.median(numeric_only=True))
    
    # 编码分类变量
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 分离特征和目标
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    \"\"\"训练随机森林模型\"\"\"
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    print("\\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    importance = pd.DataFrame({{
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }}).sort_values('importance', ascending=False)
    
    return model, importance
""",
    # 设计模式
    lambda: f"""from abc import ABC, abstractmethod
from typing import List

# 策略模式
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        if len(data) <= 1:
            return data
        pivot = data[0]
        left = [x for x in data[1:] if x <= pivot]
        right = [x for x in data[1:] if x > pivot]
        return self.sort(left) + [pivot] + self.sort(right)

class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

# 观察者模式
class Observer(ABC):
    @abstractmethod
    def update(self, message: str):
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)

# 工厂模式
class Animal(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        animals = {{"dog": Dog, "cat": Cat}}
        cls = animals.get(animal_type.lower())
        if cls is None:
            raise ValueError(f"Unknown animal type: {{animal_type}}")
        return cls()
""",
    # 数据库/SQLAlchemy
    lambda: f"""from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    orders = relationship('Order', back_populates='user')

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    description = Column(Text)
    order_items = relationship('OrderItem', back_populates='product')

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    total_amount = Column(Float, default=0.0)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='orders')
    items = relationship('OrderItem', back_populates='order')

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    order = relationship('Order', back_populates='items')
    product = relationship('Product', back_populates='order_items')

# 创建数据库连接
def init_db(db_url='sqlite:///store.db'):
    engine = create_engine(db_url, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
""",
]

# ========== Java/SpringBoot 代码模板 ==========
JAVA_SAMPLES = [
    lambda: f"""package com.example.demo.service;

import com.example.demo.model.User;
import com.example.demo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class UserService {{
    
    @Autowired
    private UserRepository userRepository;
    
    public List<User> findAll() {{
        return userRepository.findAll();
    }}
    
    public Optional<User> findById(Long id) {{
        return userRepository.findById(id);
    }}
    
    public User save(User user) {{
        return userRepository.save(user);
    }}
    
    public void deleteById(Long id) {{
        userRepository.deleteById(id);
    }}
    
    public List<User> searchByName(String keyword) {{
        return userRepository.findByNameContainingIgnoreCase(keyword);
    }}
    
    public boolean existsByEmail(String email) {{
        return userRepository.existsByEmail(email);
    }}
}}
""",
    lambda: f"""package com.example.demo.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "users")
public class User {{
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false, length = 50)
    private String name;
    
    @Column(nullable = false, unique = true, length = 100)
    private String email;
    
    @Column(length = 200)
    private String password;
    
    @Enumerated(EnumType.STRING)
    private UserRole role = UserRole.USER;
    
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt = LocalDateTime.now();
    
    @PreUpdate
    protected void onUpdate() {{
        updatedAt = LocalDateTime.now();
    }}
}}

enum UserRole {{
    USER, ADMIN, MODERATOR
}}
""",
    lambda: f"""package com.example.demo.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
public class SecurityConfig {{
    
    @Bean
    public PasswordEncoder passwordEncoder() {{
        return new BCryptPasswordEncoder();
    }}
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {{
        http
            .csrf(csrf -> csrf.disable())
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers("/api/users/**").hasAnyRole("USER", "ADMIN")
                .anyRequest().authenticated()
            )
            .sessionManagement(session -> session
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            );
        return http.build();
    }}
}}
""",
    lambda: f"""package com.example.demo.controller;

import com.example.demo.dto.LoginRequest;
import com.example.demo.dto.LoginResponse;
import com.example.demo.dto.RegisterRequest;
import com.example.demo.service.AuthService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {{
    
    @Autowired
    private AuthService authService;
    
    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest request) {{
        try {{
            authService.register(request);
            return ResponseEntity.ok(new MessageResponse("User registered successfully"));
        }} catch (Exception e) {{
            return ResponseEntity.badRequest()
                .body(new MessageResponse("Error: " + e.getMessage()));
        }}
    }}
    
    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(@Valid @RequestBody LoginRequest request) {{
        LoginResponse response = authService.login(request);
        return ResponseEntity.ok(response);
    }}
    
    @PostMapping("/logout")
    public ResponseEntity<?> logout() {{
        return ResponseEntity.ok(new MessageResponse("Logged out successfully"));
    }}
}}
""",
    lambda: f"""package com.example.demo.repository;

import com.example.demo.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {{
    
    Optional<User> findByEmail(String email);
    
    boolean existsByEmail(String email);
    
    List<User> findByNameContainingIgnoreCase(String keyword);
    
    @Query("SELECT u FROM User u WHERE u.role = :role ORDER BY u.createdAt DESC")
    List<User> findByRoleOrderByCreatedAtDesc(@Param("role") String role);
    
    @Query(value = "SELECT * FROM users WHERE created_at >= :since", nativeQuery = true)
    List<User> findRecentUsers(@Param("since") LocalDateTime since);
    
    long countByRole(String role);
}}
""",
    lambda: f"""package com.example.demo.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class RegisterRequest {{
    
    @NotBlank(message = "Name is required")
    @Size(min = 2, max = 50, message = "Name must be 2-50 characters")
    private String name;
    
    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    private String email;
    
    @NotBlank(message = "Password is required")
    @Size(min = 6, max = 100, message = "Password must be 6-100 characters")
    private String password;
}}

@Data
class LoginRequest {{
    @NotBlank(message = "Email is required")
    private String email;
    
    @NotBlank(message = "Password is required")
    private String password;
}}

@Data
class LoginResponse {{
    private String token;
    private String tokenType = "Bearer";
    private String email;
    private String role;
}}

@Data
class MessageResponse {{
    private String message;
    
    public MessageResponse(String message) {{
        this.message = message;
    }}
}}
""",
    lambda: f"""package com.example.demo.service;

import com.example.demo.exception.ResourceNotFoundException;
import com.example.demo.exception.BusinessException;
import com.example.demo.model.Order;
import com.example.demo.model.OrderStatus;
import com.example.demo.repository.OrderRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Service
@Transactional
public class OrderService {{
    
    private static final Logger log = LoggerFactory.getLogger(OrderService.class);
    
    @Autowired
    private OrderRepository orderRepository;
    
    public Order createOrder(Order order) {{
        if (order.getItems() == null || order.getItems().isEmpty()) {{
            throw new BusinessException("Order must have at least one item");
        }}
        order.setStatus(OrderStatus.PENDING);
        order.setCreatedAt(LocalDateTime.now());
        BigDecimal total = order.getItems().stream()
            .map(item -> item.getPrice().multiply(BigDecimal.valueOf(item.getQuantity())))
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        order.setTotalAmount(total);
        log.info("Creating order with total: {{}}", total);
        return orderRepository.save(order);
    }}
    
    public Page<Order> getOrdersByUser(Long userId, Pageable pageable) {{
        return orderRepository.findByUserId(userId, pageable);
    }}
    
    public void cancelOrder(Long orderId) {{
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new ResourceNotFoundException("Order not found: " + orderId));
        if (order.getStatus() == OrderStatus.SHIPPED) {{
            throw new BusinessException("Cannot cancel shipped order");
        }}
        order.setStatus(OrderStatus.CANCELLED);
        orderRepository.save(order);
        log.warn("Order {{}} cancelled", orderId);
    }}
}}
""",
]

# ========== Markdown/技术文档 ==========
MD_SAMPLES = [
    lambda: f"""# Python 性能优化指南

## 1. 使用内置函数和数据结构
Python 的内置函数是用 C 实现的，比 Python 循环快得多。

```python
# 慢
squares = []
for i in range(1000):
    squares.append(i ** 2)

# 快
squares = [i ** 2 for i in range(1000)]

# 更快
import numpy as np
squares = np.arange(1000) ** 2
```

## 2. 使用 locals() 和局部变量
局部变量访问比全局变量快。

```python
# 慢
def slow():
    for i in range(1000):
        math.sqrt(i)

# 快
def fast():
    sqrt = math.sqrt
    for i in range(1000):
        sqrt(i)
```

## 3. 字符串连接
```python
# 慢 (O(n²))
s = ''
for chunk in chunks:
    s += chunk

# 快 (O(n))
s = ''.join(chunks)
```

## 4. 使用生成器节省内存
```python
# 占用 O(n) 内存
result = [x * 2 for x in range(1000000)]

# 只占用 O(1) 内存
result = (x * 2 for x in range(1000000))
```

## 5. 使用适当的数据结构
- 查重用 set，不用 list
- 字典比 if-elif 链快
- collections.deque 比 list 做队列快
- heapq 比手动维护排序列表快
""",
    lambda: f"""# Git 工作流最佳实践

## 分支策略

### 1. Git Flow
- `main` - 生产分支
- `develop` - 开发主分支
- `feature/*` - 功能分支
- `release/*` - 发布分支
- `hotfix/*` - 紧急修复

### 2. GitHub Flow (简单)
- `main` - 生产分支
- `feature/*` - 功能分支 → PR → main

## 提交规范 (Conventional Commits)

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试
chore: 构建/工具
perf: 性能优化
```

## 分支命名

```
feature/issue-123-add-login
bugfix/issue-456-fix-npe
hotfix/1.2.3-security-patch
release/2.0.0
```

## Code Review 清单

- [ ] 代码能编译通过
- [ ] 测试覆盖率达 80%+
- [ ] 没有硬编码的配置
- [ ] 错误处理正确
- [ ] 日志记录完善
- [ ] 没有安全漏洞（SQL注入、XSS等）
- [ ] API 兼容性
- [ ] 性能影响评估
""",
    lambda: f"""# REST API 设计规范

## URL 设计

- 使用名词复数: `/api/users` 而非 `/api/getUser`
- 用 HTTP 方法表达操作: GET/POST/PUT/DELETE
- 用查询参数过滤: `?status=active&page=1&size=20`
- 版本控制: `/api/v1/users`

## 状态码使用

| 状态码 | 含义 | 场景 |
|--------|------|------|
| 200 | OK | GET 成功 |
| 201 | Created | POST 创建成功 |
| 204 | No Content | DELETE 成功 |
| 400 | Bad Request | 参数校验失败 |
| 401 | Unauthorized | 未认证 |
| 403 | Forbidden | 无权限 |
| 404 | Not Found | 资源不存在 |
| 409 | Conflict | 资源冲突 |
| 422 | Unprocessable | 业务校验失败 |
| 429 | Too Many | 限流 |
| 500 | Internal Error | 服务器异常 |

## 响应格式

```json
{{
    "code": 0,
    "message": "success",
    "data": {{ ... }},
    "pagination": {{
        "page": 1,
        "size": 20,
        "total": 100
    }}
}}
```

## 错误响应

```json
{{
    "code": 40001,
    "message": "Validation failed",
    "errors": [
        {{"field": "email", "message": "Invalid email format"}},
        {{"field": "password", "message": "Password too short"}}
    ]
}}
```
""",
]

# ========== YAML/JSON/XML 配置 ==========
YAML_SAMPLES = [
    lambda: f"""# Docker Compose 配置
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: my-postgres
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app_user -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: my-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    networks:
      - app-network

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: my-app
    ports:
      - "8080:8080"
    environment:
      SPRING_DATASOURCE_URL: jdbc:postgresql://postgres:5432/myapp
      SPRING_REDIS_HOST: redis
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:

networks:
  app-network:
    driver: bridge
""",
    lambda: f"""# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: production
  labels:
    app: my-app
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: myregistry/my-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /actuator/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: 8080
          initialDelaySeconds: 20
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: my-app
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
""",
]

# ========== 全部语料 ==========
def generate_variation(template_func, variation_id):
    """为模板生成带小幅变化的变体"""
    text = template_func()
    # 添加随机注释前缀增加多样性
    comments = [
        f"# Module: code_generator_{variation_id}\n# Author: ai_assistant\n# Description: Auto-generated code sample\n\n",
        f"# -*- coding: utf-8 -*-\n# {variation_id}: Generated code example\n\n",
        f"# =========================================\n# Code Sample {variation_id}\n# =========================================\n\n",
        f"# Copyright (c) 2024. All rights reserved.\n# File: sample_{variation_id}.py\n\n",
        "",
    ]
    prefix = random.choice(comments)
    return prefix + text


def generate_corpus():
    texts = []
    target_total_mb = 15  # 目标 ~15MB（留余量，因为还有后续扩充）
    total_size = 0
    sample_count = 0
    
    all_templates = PYTHON_SAMPLES + JAVA_SAMPLES + MD_SAMPLES + YAML_SAMPLES
    
    # 生成变体直到达到目标大小
    iteration = 0
    while total_size < target_total_mb * 1024 * 1024:
        for template_func in all_templates:
            text = generate_variation(template_func, iteration)
            texts.append({"text": text, "size": len(text.encode("utf-8"))})
            total_size += len(text.encode("utf-8"))
            sample_count += 1
            
            if total_size >= target_total_mb * 1024 * 1024:
                break
        iteration += 1
        if iteration % 10 == 0:
            print(f"  生成中... {sample_count} 条, {total_size/1024/1024:.1f}MB")
    
    print(f"生成 {sample_count} 条语料，总大小 {total_size/1024/1024:.1f}MB")
    return texts


def save_data(texts, tokenizer):
    """保存为训练集和验证集"""
    random.shuffle(texts)

    split_idx = int(len(texts) * 0.95)
    train_texts = texts[:split_idx]
    valid_texts = texts[split_idx:]

    # 保存 train
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_texts:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

    # 保存 validation
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")
    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_texts:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")

    # 统计 tokens
    train_tokens = sum(len(tokenizer.encode(t["text"])) for t in train_texts)
    valid_tokens = sum(len(tokenizer.encode(t["text"])) for t in valid_texts)

    print(f"训练集: {len(train_texts)} 条, {train_tokens} tokens, "
          f"{sum(t['size'] for t in train_texts)/1024/1024:.1f}MB")
    print(f"验证集: {len(valid_texts)} 条, {valid_tokens} tokens, "
          f"{sum(t['size'] for t in valid_texts)/1024/1024:.1f}MB")

    # 写入统计信息
    stats = {
        "train_count": len(train_texts),
        "valid_count": len(valid_texts),
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "train_size_mb": sum(t['size'] for t in train_texts) / 1024 / 1024,
        "valid_size_mb": sum(t['size'] for t in valid_texts) / 1024 / 1024,
    }
    with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    texts = generate_corpus()
    save_data(texts, tokenizer)
    print("语料准备完成！")
