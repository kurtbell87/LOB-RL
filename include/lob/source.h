#pragma once
#include "lob/message.h"

namespace lob {

class IMessageSource {
public:
    virtual ~IMessageSource() = default;
    virtual bool has_next() const = 0;
    virtual MBOMessage next() = 0;
    virtual void reset() = 0;
    virtual uint64_t message_count() const = 0;
};

}  // namespace lob
