
# Repository: projects/scml
echo -e "${BLUE}Restoring: projects/scml${NC}"
if [ -d "projects/scml/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"

    # Clone the repository
    if git clone "git@github.com:yasserfarouk/scml.git" "projects/scml"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"

        # Checkout the original branch if not already on it
        cd "projects/scml"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "master" ]; then
            if git checkout "master" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: master"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: master"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

# Repository: projects/scml
echo -e "${BLUE}Restoring: projects/scml${NC}"
if [ -d "projects/scml/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/scml.git" "projects/scml"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "projects/scml"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "master" ]; then
            if git checkout "master" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: master"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: master"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

