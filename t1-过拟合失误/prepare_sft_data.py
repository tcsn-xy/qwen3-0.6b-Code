"""
第三步：SFT 问答数据准备
生成 100-300 条代码问答数据（ChatML 格式）
"""

import json
import os
import random

OUTPUT_DIR = "data/sft"
SEED = 42
random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 代码生成问题 ==========
CODE_GENERATION = [
    {
        "user": "用 Python 写一个快速排序函数",
        "assistant": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"
    },
    {
        "user": "用 Python 实现二分查找",
        "assistant": "```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"
    },
    {
        "user": "用 Java 写一个单例模式（线程安全）",
        "assistant": "```java\npublic class Singleton {\n    private static volatile Singleton instance;\n    \n    private Singleton() {}\n    \n    public static Singleton getInstance() {\n        if (instance == null) {\n            synchronized (Singleton.class) {\n                if (instance == null) {\n                    instance = new Singleton();\n                }\n            }\n        }\n        return instance;\n    }\n}\n```\n这是双重检查锁定（Double-Checked Locking）的单例模式，volatile 关键字防止指令重排序。"
    },
    {
        "user": "写一个 SpringBoot REST 控制器的示例",
        "assistant": "```java\n@RestController\n@RequestMapping(\"/api/users\")\npublic class UserController {\n    @Autowired\n    private UserService userService;\n    \n    @GetMapping\n    public ResponseEntity<List<User>> getAllUsers() {\n        return ResponseEntity.ok(userService.findAll());\n    }\n    \n    @GetMapping(\"/{id}\")\n    public ResponseEntity<User> getUserById(@PathVariable Long id) {\n        return ResponseEntity.ok(userService.findById(id));\n    }\n    \n    @PostMapping\n    public ResponseEntity<User> createUser(@RequestBody @Valid User user) {\n        return ResponseEntity.status(HttpStatus.CREATED).body(userService.save(user));\n    }\n    \n    @DeleteMapping(\"/{id}\")\n    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {\n        userService.deleteById(id);\n        return ResponseEntity.noContent().build();\n    }\n}\n```"
    },
    {
        "user": "写一个 Python 装饰器，用于测量函数执行时间",
        "assistant": "```python\nimport time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        result = func(*args, **kwargs)\n        elapsed = time.perf_counter() - start\n        print(f\"{func.__name__} 耗时: {elapsed:.4f}s\")\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n    return \"done\"\n```"
    },
    {
        "user": "写一个 SQL 查询：查找每个部门的最高薪资员工",
        "assistant": "```sql\nSELECT d.department_name, e.name, e.salary\nFROM employees e\nJOIN departments d ON e.department_id = d.id\nWHERE e.salary = (\n    SELECT MAX(e2.salary)\n    FROM employees e2\n    WHERE e2.department_id = e.department_id\n);\n```\n或者用窗口函数更高效：\n```sql\nSELECT department_name, name, salary\nFROM (\n    SELECT d.department_name, e.name, e.salary,\n           ROW_NUMBER() OVER (PARTITION BY e.department_id ORDER BY e.salary DESC) AS rn\n    FROM employees e\n    JOIN departments d ON e.department_id = d.id\n) ranked\nWHERE rn = 1;\n```"
    },
    {
        "user": "用 Python 实现一个简单的 LRU 缓存",
        "assistant": "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    \n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    \n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```"
    },
    {
        "user": "用 Java Stream API 对一个列表进行分组、过滤和聚合",
        "assistant": "```java\n// 按分类统计销售额 > 100 的商品总数\nMap<String, Long> result = products.stream()\n    .filter(p -> p.getPrice() > 100)\n    .collect(Collectors.groupingBy(\n        Product::getCategory,\n        Collectors.counting()\n    ));\n\n// 按分类汇总销售额\nMap<String, Double> salesByCategory = orders.stream()\n    .collect(Collectors.groupingBy(\n        Order::getCategory,\n        Collectors.summingDouble(Order::getAmount)\n    ));\n```"
    },
    {
        "user": "写一个 Python 生成器，生成斐波那契数列",
        "assistant": "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\n# 使用\nfor num in fibonacci(10):\n    print(num)\n# 输出: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34\n```"
    },
    {
        "user": "写一个 SpringBoot 的全局异常处理器",
        "assistant": "```java\n@RestControllerAdvice\npublic class GlobalExceptionHandler {\n    \n    @ExceptionHandler(ResourceNotFoundException.class)\n    @ResponseStatus(HttpStatus.NOT_FOUND)\n    public ErrorResponse handleNotFound(ResourceNotFoundException ex) {\n        return new ErrorResponse(\"NOT_FOUND\", ex.getMessage());\n    }\n    \n    @ExceptionHandler(MethodArgumentNotValidException.class)\n    @ResponseStatus(HttpStatus.BAD_REQUEST)\n    public ErrorResponse handleValidation(MethodArgumentNotValidException ex) {\n        String msg = ex.getBindingResult().getFieldErrors().stream()\n            .map(e -> e.getField() + \": \" + e.getDefaultMessage())\n            .collect(Collectors.joining(\", \"));\n        return new ErrorResponse(\"VALIDATION_ERROR\", msg);\n    }\n    \n    @ExceptionHandler(Exception.class)\n    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)\n    public ErrorResponse handleGeneral(Exception ex) {\n        return new ErrorResponse(\"INTERNAL_ERROR\", \"服务器内部错误\");\n    }\n}\n\nclass ErrorResponse {\n    private final String code;\n    private final String message;\n\n    public ErrorResponse(String code, String message) {\n        this.code = code;\n        this.message = message;\n    }\n\n    public String getCode() { return code; }\n    public String getMessage() { return message; }\n}\n```"
    },
]

# ========== 代码解释问题 ==========
CODE_EXPLANATION = [
    {
        "user": "解释这段 Python 代码做了什么：\n\nresult = [x**2 for x in range(10) if x % 2 == 0]",
        "assistant": "这是一个列表推导式，做了三件事：\n1. `range(10)` 生成 0-9 的数字序列\n2. `if x % 2 == 0` 过滤出偶数（0, 2, 4, 6, 8）\n3. `x**2` 对每个偶数求平方\n\n最终结果：`[0, 4, 16, 36, 64]`\n\n等价于：\n```python\nresult = []\nfor x in range(10):\n    if x % 2 == 0:\n        result.append(x**2)\n```"
    },
    {
        "user": "解释 Java 中的 volatile 关键字的作用",
        "assistant": "`volatile` 是 Java 提供的一个轻量级同步机制，保证：\n\n**1. 可见性（Visibility）**：\n- 当一个线程修改了 volatile 变量的值，新值会立即刷新到主内存\n- 其他线程读取该变量时，会从主内存重新读取，而不是从缓存读\n\n**2. 禁止指令重排序**：\n- JVM 在编译和运行时可能会对指令重排序优化\n- volatile 变量周围的读写操作不会被重排序\n\n**不保证原子性**：\n- count++ 这种操作即使变量是 volatile 也是线程不安全的\n- 需要原子性时用 `AtomicInteger` 或 `synchronized`\n\n典型用途：双重检查锁定的单例模式中的 `instance` 变量。"
    },
    {
        "user": "解释这段代码的输出为什么是 [5, 5, 5, 5]：\n\ndef create_multipliers():\n    return [lambda x: i * x for i in range(4)]\n\nfor m in create_multipliers():\n    print(m(1))",
        "assistant": "这是 Python 闭包中的**延迟绑定（late binding）**问题。\n\n原因：\n- 列表推导式中定义的 lambda 函数捕获的是变量 `i` 的引用，而不是创建时的值\n- 当 lambda 实际执行时（调用 `m(1)`），循环已经结束，`i` 的值是最后一个值\n- 实际上 range(4) 结束后 i=3，所以每个 lambda 返回 3*1=3，输出是 [3,3,3,3]\n\n**修正方法**：\n```python\ndef create_multipliers():\n    return [lambda x, i=i: i * x for i in range(4)]  # 默认参数在定义时绑定\n```\n这样每个 lambda 的 i 默认参数在创建时就绑定了当前值。"
    },
    {
        "user": "解释 SpringBoot 的自动配置原理",
        "assistant": "SpringBoot 自动配置的核心原理：\n\n**1. @EnableAutoConfiguration**\n- 通过 `@Import(AutoConfigurationImportSelector.class)` 导入配置\n- 从 `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports` 加载所有自动配置类\n\n**2. 条件注解**\n每个自动配置类上有条件注解控制是否生效：\n- `@ConditionalOnClass`：类路径存在某个类\n- `@ConditionalOnMissingBean`：容器中没有某个 Bean\n- `@ConditionalOnProperty`：配置文件中有某属性\n- `@ConditionalOnWebApplication`：当前是 Web 应用\n\n**3. 优先级**\n- `@AutoConfigureBefore` / `@AutoConfigureAfter`：控制配置顺序\n- `@AutoConfigureOrder`：指定排序\n\n示例：\n```java\n@Configuration\n@ConditionalOnClass(DataSource.class)\n@ConditionalOnMissingBean(DataSource.class)\n@EnableConfigurationProperties(DataSourceProperties.class)\npublic class DataSourceAutoConfiguration {\n    @Bean\n    public DataSource dataSource(DataSourceProperties props) {\n        return props.initializeDataSourceBuilder().build();\n    }\n}\n```"
    },
    {
        "user": "解释 Python 中的 GIL（全局解释器锁）是什么",
        "assistant": "**GIL（Global Interpreter Lock）** 是 CPython 解释器中的一把互斥锁。\n\n**作用**：\n- 保证同一时刻只有一个线程在执行 Python 字节码\n- 简化了内存管理（特别是引用计数）的线程安全问题\n\n**影响**：\n- CPU 密集型任务：多线程反而比单线程慢（锁竞争开销）\n- I/O 密集型任务：多线程有效，因为 I/O 操作会释放 GIL\n\n**解决方案**：\n1. `multiprocessing`：多进程，每个进程有自己的 GIL\n2. 使用 C 扩展（如 NumPy）在 C 层面释放 GIL\n3. 用 asyncio 协程替代线程\n4. 使用 Jython / IronPython（没有 GIL，但生态不完整）\n\nPython 3.13 推出了**自由线程模式（Free-Threaded Python）**，可以禁用 GIL，但目前性能有损耗。"
    },
    {
        "user": "解释这段 Java 代码的作用：\n\nList<String> list = Arrays.asList(\"a\", \"bb\", \"ccc\", \"dd\");\nlist.stream()\n    .filter(s -> s.length() > 1)\n    .map(String::toUpperCase)\n    .collect(Collectors.toList());",
        "assistant": "这段代码使用了 Java Stream API 进行链式操作：\n\n1. `Arrays.asList(...)` 创建一个包含 4 个字符串的列表\n2. `.stream()` 将列表转换为流\n3. `.filter(s -> s.length() > 1)` 过滤出长度大于 1 的字符串 → 保留 \"bb\", \"ccc\", \"dd\"（\"a\" 被过滤掉）\n4. `.map(String::toUpperCase)` 将每个字符串转为大写 → \"BB\", \"CCC\", \"DD\"\n5. `.collect(Collectors.toList())` 将结果收集为新的 List\n\n最终结果：`[\"BB\", \"CCC\", \"DD\"]`"
    },
]

# ========== Bug 修复问题 ==========
BUG_FIX = [
    {
        "user": "这段代码有什么问题？怎么修复？\n\ndef append_to_list(value, my_list=[]):\n    my_list.append(value)\n    return my_list",
        "assistant": "**问题**：可变默认参数陷阱（Mutable Default Argument）。\n\nPython 的函数默认参数在**定义时**只创建一次，后续调用会复用同一个列表对象。\n\n```python\nprint(append_to_list(1))  # [1]\nprint(append_to_list(2))  # [1, 2]  ← 不是预期的 [2]\nprint(append_to_list(3))  # [1, 2, 3]\n```\n\n**修复**：\n```python\ndef append_to_list(value, my_list=None):\n    if my_list is None:\n        my_list = []\n    my_list.append(value)\n    return my_list\n```\n这样每次调用都会创建新的列表。"
    },
    {
        "user": "修复这个 Java NullPointerException 问题：\n\npublic String getCityName(User user) {\n    return user.getAddress().getCity().getName();\n}",
        "assistant": "**问题**：链式调用中任一环节可能为 null（User、Address、City 都可能为 null）。\n\n**方案 1：Java 8+ Optional**\n```java\npublic String getCityName(User user) {\n    return Optional.ofNullable(user)\n        .map(User::getAddress)\n        .map(Address::getCity)\n        .map(City::getName)\n        .orElse(\"Unknown\");\n}\n```\n\n**方案 2：传统 null 检查**\n```java\npublic String getCityName(User user) {\n    if (user == null || user.getAddress() == null \n        || user.getAddress().getCity() == null) {\n        return \"Unknown\";\n    }\n    return user.getAddress().getCity().getName();\n}\n```\n\n推荐使用 Optional 方式，代码更简洁可读。"
    },
    {
        "user": "这段 SQL 有什么性能问题？怎么优化？\n\nSELECT * FROM orders o\nWHERE o.customer_id IN (\n    SELECT id FROM customers\n    WHERE signup_date > '2024-01-01'\n)\nORDER BY o.created_at DESC",
        "assistant": "**问题**：使用 `IN (SELECT ...)` 可能导致子查询全表扫描，特别是 `customers` 和 `orders` 都很大时。\n\n**优化方案 1：使用 JOIN**\n```sql\nSELECT o.*\nFROM orders o\nJOIN customers c ON o.customer_id = c.id\nWHERE c.signup_date > '2024-01-01'\nORDER BY o.created_at DESC;\n```\n\n**优化方案 2：使用 EXISTS**（某些数据库中比 JOIN 更快）\n```sql\nSELECT o.*\nFROM orders o\nWHERE EXISTS (\n    SELECT 1 FROM customers c\n    WHERE c.id = o.customer_id\n    AND c.signup_date > '2024-01-01'\n)\nORDER BY o.created_at DESC;\n```\n\n**额外优化**：\n- `customers.signup_date` 加索引\n- `orders.customer_id` 加索引\n- `orders.created_at` 加索引\n- 避免 `SELECT *`，只选需要的列\n- 如果只要前 N 条，加 `LIMIT`"
    },
    {
        "user": "这段代码为什么会死锁？\n\nimport threading\n\nlock1 = threading.Lock()\nlock2 = threading.Lock()\n\ndef thread_a():\n    with lock1:\n        with lock2:\n            print(\"A\")\n\ndef thread_b():\n    with lock2:\n        with lock1:\n            print(\"B\")",
        "assistant": "**问题**：**死锁（Deadlock）**。两个线程以不同顺序获取锁：\n- 线程 A：先 lock1 再 lock2\n- 线程 B：先 lock2 再 lock1\n\n当 A 获取了 lock1，B 获取了 lock2 后，A 等待 lock2（B 持有），B 等待 lock1（A 持有），双方永远等待。\n\n**修复方案**：\n\n**方案 1：统一锁顺序**\n```python\ndef thread_a():\n    with lock1:\n        with lock2:\n            print(\"A\")\n\ndef thread_b():\n    with lock1:  # 同样的顺序\n        with lock2:\n            print(\"B\")\n```\n\n**方案 2：使用 threading.RLock**（可重入锁）如果确实需要嵌套\n\n**方案 3：总是以相同的全局顺序获取锁**（例如按 id 排序）"
    },
    {
        "user": "SpringBoot 中 @Autowired 注入为 null 可能是什么原因？",
        "assistant": "`@Autowired` 注入为 null 的常见原因：\n\n**1. 对象不是由 Spring 管理的**\n- 用 `new` 关键字手动创建的对象\n- 解决方法：改为 `@Component` / `@Service` 等注解，或通过配置类创建 Bean\n\n**2. 注解未被扫描**\n- 启动类上没有 `@ComponentScan` 或扫描路径不正确\n- 解决方法：确认启动类能扫描到 Bean 所在的包\n\n**3. 循环依赖**\n```\nA → B → A\n```\n- Spring 无法决定先创建哪个 Bean\n- 解决方法：用 `@Lazy` 或重构设计\n\n**4. 配置类没生效**\n- `@Configuration` 类被遗漏\n\n**5. 多例模式下没注入**\n- `@Scope(\"prototype\")` 的 Bean 需要手动获取\n\n**调试技巧**：\n- 检查启动日志是否有相关 Bean 的创建信息\n- 使用 `ApplicationContext.getBean()` 手动获取测试"
    },
]

# ========== 代码优化问题 ==========
CODE_OPTIMIZATION = [
    {
        "user": "优化这个 N+1 查询问题的 JPA 代码：\n\n@OneToMany\nprivate List<Order> orders;\n\n// 在 Service 中\nfor (User user : userRepository.findAll()) {\n    System.out.println(user.getOrders().size());\n}",
        "assistant": "**问题**：N+1 查询。1 次查询用户列表 + N 次查询每个用户的订单。\n\n**方案 1：JOIN FETCH**\n```java\n@Query(\"SELECT DISTINCT u FROM User u LEFT JOIN FETCH u.orders\")\nList<User> findAllWithOrders();\n```\n\n**方案 2：@EntityGraph**\n```java\n@EntityGraph(attributePaths = \"orders\")\n@Query(\"SELECT u FROM User u\")\nList<User> findAllWithOrders();\n```\n\n**方案 3：批量查询**\n```java\n// 先查所有用户，再批量查订单\nList<User> users = userRepository.findAll();\nList<Long> userIds = users.stream().map(User::getId).toList();\nMap<Long, List<Order>> orderMap = orderRepository.findByUserIdIn(userIds)\n    .stream()\n    .collect(Collectors.groupingBy(Order::getUserId));\n```\n\n**方案 4：Hibernate 批量设置**\n```yaml\nspring.jpa.properties.hibernate.default_batch_fetch_size=50\n```\n推荐使用 `JOIN FETCH` 或 `@EntityGraph`，最直接有效。"
    },
    {
        "user": "优化这段 Python 代码的性能：\n\nresult = []\nfor i in range(len(items)):\n    for j in range(len(items)):\n        if items[i] == items[j] and i != j:\n            result.append(items[i])",
        "assistant": "**问题**：O(n²) 的双重循环，检查是否有重复元素，但逻辑有误（会重复添加）。\n\n**优化方案 1：使用集合（O(n)）**\n```python\ndef find_duplicates(items):\n    seen = set()\n    duplicates = set()\n    for item in items:\n        if item in seen:\n            duplicates.add(item)\n        else:\n            seen.add(item)\n    return list(duplicates)\n```\n\n**优化方案 2：使用 Counter**\n```python\nfrom collections import Counter\n\ndef find_duplicates(items):\n    return [item for item, count in Counter(items).items() if count > 1]\n```\n\n**对比**：\n- 原代码：O(n²)，n=1000 时约 1,000,000 次比较\n- 集合方案：O(n)，n=1000 时约 1000 次操作，快 1000 倍"
    },
]

# ========== 技术问答 ==========
TECH_QA = [
    {
        "user": "Python 中 __init__ 和 __new__ 的区别是什么？",
        "assistant": "**`__new__`** 和 **`__init__`** 是 Python 对象创建的两个步骤：\n\n| 方法 | 作用 | 调用时机 | 返回值 |\n|------|------|---------|--------|\n| `__new__` | 创建并返回新对象 | 最先调用 | 对象实例 |\n| `__init__` | 初始化已创建的对象 | 在 __new__ 之后 | None |\n\n```python\nclass MyClass:\n    def __new__(cls, *args, **kwargs):\n        print(\"1. 创建实例\")\n        instance = super().__new__(cls)\n        return instance\n    \n    def __init__(self, value):\n        print(\"2. 初始化实例\")\n        self.value = value\n\nobj = MyClass(42)\n# 输出:\n# 1. 创建实例\n# 2. 初始化实例\n```\n\n**使用场景**：\n- `__new__`：实现单例模式、不可变类型（如 tuple 的子类）\n- `__init__`：常规初始化，99% 的情况用这个就够了"
    },
    {
        "user": "Java 中 HashMap 和 ConcurrentHashMap 的区别",
        "assistant": "**HashMap vs ConcurrentHashMap 核心区别**：\n\n| 特性 | HashMap | ConcurrentHashMap |\n|------|---------|-------------------|\n| 线程安全 | ❌ 否 | ✅ 是 |\n| 允许 null key | ✅ 是 | ❌ 否 |\n| 性能（单线程） | 高 | 略低（CAS 开销） |\n| 性能（多线程） | 危险 | 高（分段锁/红黑树优化） |\n\n**ConcurrentHashMap 实现原理（Java 8+）**：\n1. 使用 **CAS + synchronized** 控制并发\n2. 对每个桶单独加锁，不锁整个表\n3. 扩容时允许多线程协助迁移\n4. 使用红黑树优化哈希冲突时的查询性能\n\n**使用建议**：\n- 单线程环境：`HashMap`\n- 多线程只读：`HashMap` + `Collections.unmodifiableMap()`\n- 多线程读写：`ConcurrentHashMap`"
    },
    {
        "user": "SpringBoot 和 Spring Framework 的区别是什么？",
        "assistant": "**SpringBoot 是 Spring Framework 的扩展**，核心区别：\n\n| 维度 | Spring Framework | SpringBoot |\n|------|-----------------|------------|\n| 配置方式 | XML / Java Config 手动配置 | 自动配置（Auto Configuration） |\n| 启动方式 | Web 服务器需额外配置 | 内嵌 Tomcat/Jetty，main() 启动 |\n| 依赖管理 | 手动管理版本 | Starter POM 统一管理 |\n| 监控 | 需额外集成 | Actuator 开箱即用 |\n| 打包 | WAR 部署到外部容器 | JAR 直接运行 |\n\n**本质**：SpringBoot = Spring Framework + Auto Configuration + Embedded Server + Actuator\n\nSpringBoot 并不是替代 Spring，而是简化了 Spring 的使用。"
    },
    {
        "user": "Python 的上下文管理器（with 语句）是如何工作的？",
        "assistant": "上下文管理器通过 `__enter__` 和 `__exit__` 方法实现。\n\n**基本实现**：\n```python\nclass FileManager:\n    def __init__(self, filename, mode):\n        self.filename = filename\n        self.mode = mode\n    \n    def __enter__(self):\n        self.file = open(self.filename, self.mode)\n        return self.file  # 返回值赋给 as 后的变量\n    \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.file.close()\n        # 返回 True 会抑制异常\n        return False\n\nwith FileManager(\"test.txt\", \"r\") as f:\n    print(f.read())\n```\n\n**contextlib 简化写法**：\n```python\nfrom contextlib import contextmanager\n\n@contextmanager\ndef file_manager(filename, mode):\n    f = open(filename, mode)\n    try:\n        yield f  # with 块中 as 的绑定对象\n    finally:\n        f.close()\n```\n\n`__exit__` 方法的三个参数：`exc_type`（异常类型）、`exc_val`（异常实例）、`exc_tb`（回溯信息）。如果没有异常，三个都是 None。"
    },
    {
        "user": "Java 中 wait() 和 sleep() 的区别是什么？",
        "assistant": "| 特性 | wait() | sleep() |\n|------|--------|---------|\n| 所属类 | Object | Thread |\n| 是否释放锁 | ✅ 是，释放监视器锁 | ❌ 否，不释放锁 |\n| 调用前提 | 必须在 synchronized 块中 | 任何地方 |\n| 唤醒方式 | notify() / notifyAll() / 超时 | 超时自动唤醒 |\n| 用途 | 线程间通信协调 | 暂停当前线程执行 |\n\n```java\n// wait/notify 示例\nsynchronized (sharedObject) {\n    while (!condition) {\n        sharedObject.wait();  // 释放锁，等待条件满足\n    }\n    // 条件满足后的操作\n}\n\n// sleep 示例\nThread.sleep(1000);  // 线程暂停 1 秒，不释放锁\n```"
    },
]

# ========== 文档/README 类 ==========
README_TEMPLATES = [
    {
        "user": "帮我写一个 Python 项目的 README.md 模板",
        "assistant": "# Project Name\n\n## 简介\n一句话描述项目做什么。\n\n## 功能特性\n- ✅ 功能 1：简要说明\n- ✅ 功能 2：简要说明\n- ✅ 功能 3：简要说明\n\n## 环境要求\n- Python >= 3.8\n- pip >= 21.0\n\n## 快速开始\n\n### 安装\n```bash\ngit clone https://github.com/user/project.git\ncd project\npip install -r requirements.txt\n```\n\n### 使用\n```python\nfrom project import main\nmain.run()\n```\n\n## 配置\n| 参数 | 类型 | 默认值 | 说明 |\n|------|------|--------|------|\n| `DEBUG` | bool | False | 调试模式 |\n| `PORT` | int | 8080 | 服务端口 |\n\n## 测试\n```bash\npytest tests/ -v\n```\n\n## 项目结构\n```\nproject/\n├── src/          # 源代码\n├── tests/        # 测试\n├── docs/         # 文档\n└── config/       # 配置\n```\n\n## 贡献指南\n1. Fork 本仓库\n2. 创建特性分支 (`git checkout -b feature/amazing-feature`)\n3. 提交改动 (`git commit -m 'Add amazing feature'`)\n4. 推送到分支 (`git push origin feature/amazing-feature`)\n5. 提交 Pull Request\n\n## License\nMIT"
    },
    {
        "user": "写一个 SpringBoot 项目的 application.yml 配置文件模板",
        "assistant": "```yaml\nserver:\n  port: 8080\n  servlet:\n    context-path: /api\n\nspring:\n  application:\n    name: my-springboot-app\n\n  # 数据源配置\n  datasource:\n    url: jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=Asia/Shanghai\n    username: root\n    password: your_password\n    driver-class-name: com.mysql.cj.jdbc.Driver\n    hikari:\n      maximum-pool-size: 20\n      minimum-idle: 5\n      idle-timeout: 300000\n\n  # JPA 配置\n  jpa:\n    hibernate:\n      ddl-auto: update\n    show-sql: true\n    properties:\n      hibernate:\n        format_sql: true\n        dialect: org.hibernate.dialect.MySQL8Dialect\n\n# 日志配置\nlogging:\n  level:\n    root: INFO\n    com.mycompany: DEBUG\n  file:\n    name: logs/app.log\n    max-size: 10MB\n    max-history: 7\n```"
    },
]

# ========== 组装 ==========
def build_sft_data():
    data = []

    categories = [
        ("代码生成", CODE_GENERATION),
        ("代码解释", CODE_EXPLANATION),
        ("Bug修复", BUG_FIX),
        ("代码优化", CODE_OPTIMIZATION),
        ("技术问答", TECH_QA),
        ("文档模板", README_TEMPLATES),
    ]

    for cat_name, items in categories:
        for item in items:
            data.append({
                "messages": [
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]},
                ],
                "category": cat_name,
            })

    # 合并补充数据
    try:
        from prepare_sft_data_extra import get_extra_data
        extra = get_extra_data()
        print(f"  已加载补充数据: {len(extra)} 条")
        data.extend(extra)
    except ImportError:
        print("  (未找到补充数据模块，使用基础数据)")

    return data


def main():
    data = build_sft_data()
    print(f"共生成 {len(data)} 条问答数据")

    # 按类别统计
    from collections import Counter
    cat_count = Counter(d["category"] for d in data)
    for cat, count in cat_count.most_common():
        print(f"  {cat}: {count} 条")

    # 切分训练/验证
    random.seed(SEED)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    # 保存
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    valid_path = os.path.join(OUTPUT_DIR, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_path, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n训练集: {len(train_data)} 条 → {train_path}")
    print(f"验证集: {len(valid_data)} 条 → {valid_path}")

    # 统计 tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    total_tokens = 0
    for item in data:
        text = tokenizer.apply_chat_template(item["messages"], tokenize=True)
        total_tokens += len(text)

    print(f"总 tokens 数（含模板）: {total_tokens}")
    print(f"平均每条 tokens: {total_tokens // len(data)}")


if __name__ == "__main__":
    main()
