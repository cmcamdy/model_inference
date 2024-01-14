

### 在VSCode中为代码添加注释、文件头、代码块分割线
- https://zhuanlan.zhihu.com/p/647274361

- 在settings.json中设置doxdocgen，就可以获得
```cpp
/**
 * ************************************************************
 * @file          reduce_sum.cc
 * @author        cmcandy
 * @brief 
 * @version       0.1
 * @date          YYYY-MM-DD
 * @copyright     Copyright (c) 2024
 * ************************************************************
 */
```

```json
    "doxdocgen.c.firstLine": "/**",
    
    "doxdocgen.file.fileOrder": [
        "custom",
        "file",
        "author",
        "brief",
        "version",
        "date",
        "copyright",
        "custom"
    ],
    "doxdocgen.file.fileTemplate":  "@file          {name}",
    "doxdocgen.file.customTag": [
        "************************************************************"
    ],
    "doxdocgen.generic.authorTag":  "@author        cmcandy",
    "doxdocgen.generic.dateFormat": "         YYYY-MM-DD",
    "doxdocgen.file.versionTag":    "@version       0.1",
    "doxdocgen.file.copyrightTag": [
                                    "@copyright     Copyright (c) 2024"
    ],
```