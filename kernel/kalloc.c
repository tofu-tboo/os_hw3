// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"
#define PA2PGN(pa) (((uint64)(pa) - KERNBASE) / PGSIZE)
#define PA2SLOT(pa) (((uint64)(pa) - KERNBASE) / LPGSIZE)
#define SLOT2PA(slot) (KERNBASE + ((uint64)(slot) * LPGSIZE))
#define NMEGA ((((uint64)PHYSTOP - KERNBASE) + LPGSIZE - 1) / LPGSIZE)
#define NPAGES (PA2PGN(PHYSTOP))

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
  struct run *prev;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

struct slot {
  ushort refcnt;
} slots[NMEGA]; // superpage allocation slots
uint prefcnt[NPAGES];
void incref(void* pa)
{
  acquire(&kmem.lock);
  prefcnt[PA2PGN(pa)]++;
  slots[PA2SLOT(pa)].refcnt++; // slot refcnt also increases
  release(&kmem.lock);
}
void sincref(void* pa)
{
  int i;
  acquire(&kmem.lock);
  for (i = 0; i < PGPERSLOT; i++)
  {
    prefcnt[PA2PGN(pa) + i]++;
  }
  slots[PA2SLOT(pa)].refcnt += PGPERSLOT;
  release(&kmem.lock);
}
static void decref(void* pa)
{
  if (prefcnt[PA2PGN(pa)] == 0 || slots[PA2SLOT(pa)].refcnt == 0)
    panic("decref");
  prefcnt[PA2PGN(pa)]--;
  slots[PA2SLOT(pa)].refcnt--;
}
static void sdecref(void* pa)
{
  int i;
  for (i = 0; i < PGPERSLOT; i++)
  {
    if (prefcnt[PA2PGN(pa) + i] == 0)
      panic("sdecref");
    prefcnt[PA2PGN(pa) + i]--;
  }
  slots[PA2SLOT(pa)].refcnt -= PGPERSLOT;
}

void
kinit()
{
  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  struct slot *s;
  p = (char*)PGROUNDUP((uint64)pa_start);
  
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
  {
    s = &slots[PA2SLOT(p)];
    
    prefcnt[PA2PGN(p)] = 1;
    if (s->refcnt == 0)
      s->refcnt = PGPERSLOT;

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
  uint pgn = PA2PGN(pa);

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");

  acquire(&kmem.lock);
  if (prefcnt[pgn] == 0 || slots[PA2SLOT(pa)].refcnt == 0)
    panic("kfree: refcnt");
  decref(pa);
  if(prefcnt[pgn] > 0){
    release(&kmem.lock);
    return;
  }
  r = (struct run*)pa;

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r->next = kmem.freelist;
  r->prev = 0;
  if (kmem.freelist)
    kmem.freelist->prev = r;
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
    slots[PA2SLOT(r)].refcnt++;
    prefcnt[PA2PGN(r)] = 1;
    if (r->next)
      r->next->prev = 0;
    kmem.freelist = r->next;
    r->next = 0;
    r->prev = 0;
  }
  release(&kmem.lock);

  if(r)
    memset((char*)r, 5, PGSIZE); // fill with junk
  return (void*)r;
}

void
superfree(void *pa)
{
  struct run *r;
  uint slotno = PA2SLOT(pa);

  acquire(&kmem.lock);
  sdecref(pa);
  if (slots[slotno].refcnt > 0)
  {
    release(&kmem.lock);
    return;
  }

  // Fill with junk to catch dangling refs.
  memset(pa, 1, LPGSIZE);

  // re-point freelist cotiguously placed in phy mem
  uint64 pa1;
  for (pa1 = (uint64)pa; pa1 < (uint64)pa + LPGSIZE; pa1 += PGSIZE)
  {
    r = (struct run*)pa1;
    r->next = kmem.freelist;
    r->prev = 0;
    if (kmem.freelist)
      kmem.freelist->prev = r;
    kmem.freelist = r;
  }
  release(&kmem.lock);
}

void *
superalloc(void)
{
  uint i, j;
  uint pgn;
  uint64 pa0 = 0;

  acquire(&kmem.lock);
  for (i = 0; i < NMEGA; i++)
  {
    if (slots[i].refcnt == 0)
    { // found a free superpage slot
      pa0 = SLOT2PA(i);
      pgn = PA2PGN(pa0);
      for (j = 0; j < PGPERSLOT; j++)
      {
        if (prefcnt[pgn + j] != 0)
          panic("superalloc");
        prefcnt[pgn + j] = 1;
      }
      slots[i].refcnt = PGPERSLOT;
      
      // re-point freelist cotiguously placed in phy mem
      uint64 pa1;
      struct run *r;
      for (pa1 = pa0; pa1 < pa0 + LPGSIZE; pa1 += PGSIZE)
      {
        r = (struct run*)pa1;
        if(r->prev)
          r->prev->next = r->next;
        else
          kmem.freelist = r->next;

        if(r->next)
          r->next->prev = r->prev;

        r->next = 0;
        r->prev = 0;
      }
      break;
    }
  }

  if (pa0 != 0)
    memset((char*)pa0, 5, LPGSIZE); // fill with junk
  release(&kmem.lock);
  return (void*)pa0;
}
