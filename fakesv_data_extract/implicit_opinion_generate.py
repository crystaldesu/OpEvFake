import pickle
import openai
import time
from tqdm import tqdm
import concurrent.futures

# 设置DeepSeek API密钥
openai.api_key = ""  # deepseek api
openai.api_base = "https://api.deepseek.com/v1"

# 设置最大并发数（根据API限制调整）
MAX_WORKERS = 100


def load_data(filename):
    """加载测试数据"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_data(data, filename):
    """保存结果数据"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def prepare_news_input(item):
    """准备新闻输入文本，包含全部评论"""
    # 组合标题和OCR文本
    content = f"标题: {item['title']}\n"

    if item['ocr'] and item['ocr'].strip():
        content += f"OCR文本: {item['ocr']}\n"

    # 添加用户信息
    content += f"作者简介: {item['author_intro']}\n"
    content += f"作者地区: {item['author_place']}\n"

    # 添加全部评论
    if item['comments']:
        content += "用户评论:\n"
        for i, comment in enumerate(item['comments'], 1):
            content += f"{i}. {comment}\n"

    return content


def get_news_type(news_input, video_id):
    """第一步：判断新闻类型（事件型或常识型）"""
    system_prompt = "假设您是专业新闻检测专家。请分析以下短视频新闻内容，判断其属于事件型新闻还是常识型新闻。"

    user_prompt = f"""基于以下短视频新闻内容：
{news_input}

请判断内容类型：如果是事件型新闻，输出"事件"；如果是常识型新闻，输出"常识"。
只需输出一个词(不包含引号)，不要输出其他内容。"""

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        result = response.choices[0].message.content.strip()
        return result.lower() if result else "事件"  # 默认返回事件型

    except Exception as e:
        print(f"API调用出错: {e}")
        return "事件"  # 出错时默认返回事件型


def get_implicit_opinion(news_input, video_id, news_type):
    """第二步：根据新闻类型获取隐式观点"""
    system_prompt = "您是专业新闻检测专家。请基于新闻类型提供相应的可信度分析。"

    if news_type == "事件":
        user_prompt = f"""基于以下短视频新闻内容：
{news_input}

请从四个维度综合评价内容可信度的隐式观点：
1.事实准确性（高/中/低）
2.内容来源（可信/可疑/不可信）
3.证据支持（强/中/弱）
4.语言风格（恰当/夸张）

输出格式：事实准确性高/中/低,内容来源可信/可疑/不可信,证据支持强/中/弱,语言风格恰当/夸张
不要输出其他信息，严格按照输出格式输出。"""
    else:  # 常识型新闻
        user_prompt = f"""基于以下短视频新闻内容：
{news_input}

基于科学知识分析内容可信度的隐式观点。

输出格式：分析内容
注意输出的内容中不包含[分析内容]这四个字，输出的内容不要长篇大论 """

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API调用出错: {e}")
        return None


def process_single_video(item):
    """处理单个视频"""
    try:
        # 准备输入
        news_input = prepare_news_input(item)
        video_id = item['video_id']

        # 第一步：判断新闻类型
        news_type = get_news_type(news_input, video_id)

        # 第二步：获取隐式观点
        opinion = get_implicit_opinion(news_input, video_id, news_type)

        if opinion is None:
            return video_id, f"{video_id}:API调用失败"
        else:
            return video_id, opinion

    except Exception as e:
        video_id = item['video_id']
        print(f"处理视频 {video_id} 时出现异常: {e}")
        return video_id, f"{video_id}:处理异常 - {str(e)}"


def process_videos_parallel(data_list, max_workers=MAX_WORKERS):
    """并行处理所有视频数据"""
    results = {}

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_single_video, item): item for item in data_list}

        # 使用tqdm显示进度
        for future in tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(data_list),
                desc="处理视频"
        ):
            try:
                video_id, opinion = future.result()
                results[video_id] = opinion  # 总是添加到结果中
            except Exception as e:
                item = future_to_item[future]
                video_id = item['video_id']
                print(f"处理视频 {video_id} 时出错: {e}")
                results[video_id] = f"{video_id}:处理失败"

    return results


def process_videos_test(data_list, num_test=5):
    """测试处理前几条视频数据"""
    results = {}

    # 只处理前num_test条数据
    test_data = data_list[:num_test]

    # 使用并行处理
    results = process_videos_parallel(test_data, max_workers=min(MAX_WORKERS, num_test))

    # 打印处理结果
    for video_id, opinion in results.items():
        print(f"\n视频 {video_id}:")
        print(f"生成的隐式观点: {opinion}")

    return results


def main():
    # 加载测试数据
    print("加载数据...")
    test_data = load_data('../data/data_train_list.pkl')

    # 处理所有数据
    print("开始处理所有视频数据...")
    implicit_opinions = process_videos_parallel(test_data)

    # 保存结果
    print("保存结果...")
    save_data(implicit_opinions, 'implicit_opinion_train.pkl')

    print(f"完成! 共处理 {len(implicit_opinions)} 个视频")
    print("前几个结果示例:")
    for i, (video_id, opinion) in enumerate(list(implicit_opinions.items())[:3]):
        print(f"{video_id}: {opinion}")


if __name__ == "__main__":
    main()