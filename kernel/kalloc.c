// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"
#define PA2PFN(pa) (((uint64)(pa) - KERNBASE) / PGSIZE)
#define NPAGES (PA2PFN(PHYSTOP))

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

uint prefcnt[NPAGES];
void incref(void* pa)
{
  acquire(&kmem.lock);
  uint *cref = &prefcnt[PA2PFN(pa)];
  (*cref)++;
  release(&kmem.lock);
}

void
kinit()
{
  int i;
  for (i = 0; i < NPAGES; i++)
    prefcnt[i] = 0;

  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
  {
    prefcnt[PA2PFN(p)] = 1;
    kfree(p);
  }
}

// Free the page of physical memory pointed at by pa,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;
  uint pfn = PA2PFN(pa);
  uint *cref = &prefcnt[pfn];

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP || *cref == 0)
    panic("kfree");

  acquire(&kmem.lock);
  if(--(*cref) > 0){
    release(&kmem.lock);
    return;
  }
  r = (struct run*)pa;

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r)
  {
    kmem.freelist = r->next;
    prefcnt[PA2PFN(r)] = 1;
  }
  release(&kmem.lock);

  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}
