def dalvik_to_java_method(signature: str) -> str:
    s = signature.strip()
    # 检查基本格式
    if not (s.startswith('L') and ';->' in s and '(' in s):
        return None
    # 找到类结束标志
    try:
        l_start = 1  # 跳过 'L'
        l_end = s.index(';')
        class_path = s[l_start:l_end].replace('/', '.')  # 保留类名中的 $
        arrow_pos = s.index('->', l_end)
        paren_pos = s.index('(', arrow_pos)
        method_name = s[arrow_pos + 2:paren_pos]

        # 1) 将 <init> 这样的构造方法转换为 init（去掉尖括号）
        if method_name.startswith('<') and method_name.endswith('>'):
            method_name = method_name[1:-1]  # <init> -> init

        # 2) 如果方法名包含 $（内部/合成方法），截断为 $ 之前的部分
        if '$' in method_name:
            method_name = method_name.split('$', 1)[0]

        return f"{class_path}.{method_name}"
    except (ValueError, IndexError):
        return None


# 示例用法
if __name__ == "__main__":
    # 内部/合成方法：去掉 $ 之后的部分
    print(dalvik_to_java_method(
        "Landroid/support/v4/animation/DonutAnimatorCompatProvider$DonutFloatValueAnimator;->access$400(Landroid/support/v4/animation/DonutAnimatorCompatProvider$DonutFloatValueAnimator;)V"
    ))  # 输出: android.support.v4.animation.DonutAnimatorCompatProvider$DonutFloatValueAnimator.access

    # 构造方法 <init> -> init
    print(dalvik_to_java_method(
        "Lcom/example/MyClass;-><init>(I)V"
    ))  # 输出: com.example.MyClass.init

    # lambda 合成方法：lambda$do$0 -> lambda
    print(dalvik_to_java_method(
        "Lcom/example/MyClass;->lambda$do$0()V"
    ))  # 输出: com.example.MyClass.lambda
    
    print(dalvik_to_java_method(
        "Lcom/android/quicksearchbox/google/GoogleSuggestClient;->queryExternal(Ljava/lang/String;)Lcom/android/quicksearchbox/SourceResult;"
    ))  # 
    print(dalvik_to_java_method(
        "Landroid/os/Bundle;->putFloatArray(Ljava/lang/String;[F)V"
    ))  # 
    print(dalvik_to_java_method(
        "Landroid/content/ContextWrapper;->sendBroadcastAsUser(Landroid/content/Intent;Landroid/os/UserHandle;)V"
    ))  # 
    print(dalvik_to_java_method(
        "Landroid/database/sqlite/SQLiteDatabase;->rawQuery(Ljava/lang/String;[Ljava/lang/String;)Landroid/database/Cursor;"
    ))  # 
    print(dalvik_to_java_method(
        "Lcom/android/mail/browse/SpamWarningView;-><init>(Landroid/content/Context;)V"
    ))  # 
    



