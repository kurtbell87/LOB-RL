#pragma once
#include "lob/message.h"

class IMessageSource {
public:
    virtual ~IMessageSource() = default;
    virtual bool next(Message& msg) = 0;
    virtual void reset() = 0;
};
