def dalvik_to_java_method(signature: str) -> str:
    s = signature.strip()
    # Check basic format
    if not (s.startswith('L') and ';->' in s and '(' in s):
        return None
    # Find class end marker
    try:
        l_start = 1  # Skip leading 'L'
        l_end = s.index(';')
        class_path = s[l_start:l_end].replace('/', '.')  # Keep $ symbol in class name
        arrow_pos = s.index('->', l_end)
        paren_pos = s.index('(', arrow_pos)
        method_name = s[arrow_pos + 2:paren_pos]

        # 1) Convert constructors like <init> to init (remove angle brackets)
        if method_name.startswith('<') and method_name.endswith('>'):
            method_name = method_name[1:-1]  # <init> -> init

        # 2) If method name contains $, truncate before the $
        if '$' in method_name:
            method_name = method_name.split('$', 1)[0]

        return f"{class_path}.{method_name}"
    except (ValueError, IndexError):
        return None


# Example usage
if __name__ == "__main__":
    # For internal/synthetic methods: truncate after the first $
    print(dalvik_to_java_method(
        "Landroid/support/v4/animation/DonutAnimatorCompatProvider$DonutFloatValueAnimator;->access$400(Landroid/support/v4/animation/DonutAnimatorCompatProvider$DonutFloatValueAnimator;)V"
    ))  # Output: android.support.v4.animation.DonutAnimatorCompatProvider$DonutFloatValueAnimator.access

    # Constructor <init> -> init
    print(dalvik_to_java_method(
        "Lcom/example/MyClass;-><init>(I)V"
    ))  # Output: com.example.MyClass.init

    # Lambda synthetic method: lambda$do$0 -> lambda
    print(dalvik_to_java_method(
        "Lcom/example/MyClass;->lambda$do$0()V"
    ))  # Output: com.example.MyClass.lambda
    
    print(dalvik_to_java_method(
        "Lcom/android/quicksearchbox/google/GoogleSuggestClient;->queryExternal(Ljava/lang/String;)Lcom/android/quicksearchbox/SourceResult;"
    ))  
    print(dalvik_to_java_method(
        "Landroid/os/Bundle;->putFloatArray(Ljava/lang/String;[F)V"
    ))  
    print(dalvik_to_java_method(
        "Landroid/content/ContextWrapper;->sendBroadcastAsUser(Landroid/content/Intent;Landroid/os/UserHandle;)V"
    ))  
    print(dalvik_to_java_method(
        "Landroid/database/sqlite/SQLiteDatabase;->rawQuery(Ljava/lang/String;[Ljava/lang/String;)Landroid/database/Cursor;"
    ))  
    print(dalvik_to_java_method(
        "Lcom/android/mail/browse/SpamWarningView;-><init>(Landroid/content/Context;)V"
    ))  